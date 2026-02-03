# models/fnp_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# SetConv (kernel-based set convolution)
# -------------------------
class SetConv(nn.Module):
    """
    coords: (P, 3) where P=L*N, 3=(t,x,y)  (turbine-level coords)
    x:      (B, P, Din)
    out:    (B, P, Dout)
    """
    def __init__(self, din, dout):
        super().__init__()
        self.proj = nn.Linear(din, dout)
        self.log_lt = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, coords):
        B, P, _ = x.shape
        h = self.proj(x)

        t  = coords[:, 0:1]      # (P,1)
        xy = coords[:, 1:3]      # (P,2)

        dt  = t - t.t()          # (P,P)
        dxy = xy[:, None, :] - xy[None, :, :]  # (P,P,2)

        lt = torch.exp(self.log_lt) + 1e-6
        ls = torch.exp(self.log_ls) + 1e-6

        dist2 = (dt / lt) ** 2 + (dxy[...,0]/ls)**2 + (dxy[...,1]/ls)**2
        K = torch.exp(-0.5 * dist2).to(x.device)
        W = K / (K.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.einsum("pq,bqd->bpd", W, h)
        return out

# -------------------------
# Neural Fourier Layer (NFL)
# -------------------------
class NeuralFourierLayer(nn.Module):
    """
    input:  (B,N,L,D)
    output: (B,N,L,D)
    """
    def __init__(self, d_model, modes=8, kernel_size=3):
        super().__init__()
        self.d = d_model
        self.modes = modes

        self.Wr = nn.Parameter(torch.randn(modes, d_model, d_model) * 0.02)
        self.Wi = nn.Parameter(torch.randn(modes, d_model, d_model) * 0.02)

        pad = kernel_size // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=pad, groups=1)
        self.norm = nn.LayerNorm(d_model)

    def _complex_linear(self, x_fft):
        xr = x_fft.real
        xi = x_fft.imag
        K = x_fft.size(2)

        Wr = self.Wr[:K]
        Wi = self.Wi[:K]

        yr = torch.einsum("bnkd,kdf->bnkf", xr, Wr) - torch.einsum("bnkd,kdf->bnkf", xi, Wi)
        yi = torch.einsum("bnkd,kdf->bnkf", xr, Wi) + torch.einsum("bnkd,kdf->bnkf", xi, Wr)
        return torch.complex(yr, yi)

    def forward(self, x):
        B, N, L, D = x.shape
        x_fft = torch.fft.rfft(x, dim=2)
        Lf = x_fft.shape[2]
        K = min(self.modes, Lf)

        y_fft = x_fft.clone()
        y_fft[:, :, :K, :] = self._complex_linear(x_fft[:, :, :K, :]) + x_fft[:, :, :K, :]
        y = torch.fft.irfft(y_fft, n=L, dim=2)

        y2 = x.permute(0,1,3,2).reshape(B*N, D, L)
        y2 = self.conv(y2).reshape(B, N, D, L).permute(0,1,3,2)

        out = self.norm(y + y2)
        return out

# -------------------------
# Scheme A for obs (SCADA): embed -> SetConv -> NFL
# -------------------------
class FuncRepVFR(nn.Module):
    """
    x: (B,L,N,F)
    out: (B,L,N,D)
    """
    def __init__(self, in_dim, d_model, modes=8):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.setconv = SetConv(d_model, d_model)
        self.nfl = NeuralFourierLayer(d_model, modes=modes)

    def forward(self, x, coords):
        B, L, N, _ = x.shape
        h = self.in_proj(x)                          # (B,L,N,D)
        h = h.reshape(B, L*N, -1)                    # (B,P,D)
        h = self.setconv(h, coords)                  # (B,P,D)
        h = h.reshape(B, L, N, -1).permute(0,2,1,3)  # (B,N,L,D)
        h = self.nfl(h)
        h = h.permute(0,2,1,3)                       # (B,L,N,D)
        return h

# -------------------------
# NEW: local SetConv over K neighbor grid points (per (B,L,N))
# -------------------------
class BgPointSetConv(nn.Module):
    """
    p:   (B,L,N,K,D)
    pos: (B,L,N,K,3) where 3=(dx,dy,dist) normalized
    v:   (B,L,N,K)   point valid mask (0/1)

    Output:
      mixed: (B,L,N,K,D)  kernel mixing inside K
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        # separate lengthscales: xy and dist
        self.log_ls_xy = nn.Parameter(torch.tensor(0.0))
        self.log_ls_d  = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, p: torch.Tensor, pos: torch.Tensor, v: torch.Tensor = None) -> torch.Tensor:
        # p: (B,L,N,K,D), pos: (B,L,N,K,3)
        if p.ndim != 5 or pos.ndim != 5:
            raise RuntimeError(f"BgPointSetConv expects p/pos with shape (B,L,N,K,*), got p={tuple(p.shape)}, pos={tuple(pos.shape)}")
        B, L, N, K, D = p.shape
        if D != self.d_model:
            raise RuntimeError(f"d_model mismatch: p.D={D}, expected {self.d_model}")
        if pos.shape[-1] != 3:
            raise RuntimeError(f"pos last dim must be 3 (dx,dy,dist), got {pos.shape[-1]}")

        # pairwise differences within K
        # dp: (B,L,N,K,K,3)
        dp = pos.unsqueeze(-2) - pos.unsqueeze(-3)  # k_i - k_j
        ddx = dp[..., 0]
        ddy = dp[..., 1]
        dds = dp[..., 2]

        ls_xy = torch.exp(self.log_ls_xy) + 1e-6
        ls_d  = torch.exp(self.log_ls_d)  + 1e-6

        dist2 = (ddx / ls_xy) ** 2 + (ddy / ls_xy) ** 2 + (dds / ls_d) ** 2
        Kmat = torch.exp(-0.5 * dist2)  # (B,L,N,K,K)

        if v is not None:
            # v is source-point mask (j dimension)
            # v_src: (B,L,N,1,K)
            v_src = v.unsqueeze(-2)
            Kmat = Kmat * v_src

        W = Kmat / (Kmat.sum(dim=-1, keepdim=True) + 1e-6)  # normalize over j

        mixed = torch.einsum("blnkj,blnjd->blnkd", W, p)  # (B,L,N,K,D)
        # residual + norm (more stable than overwrite)
        out = self.norm(p + mixed)
        return out

# -------------------------
# BgGridEncoder for CERRA (K=16 + dx/dy/dist)
#   decode block-major -> per-point -> (optional) local SetConv in K -> attention pool
# -------------------------
class BgGridEncoder(nn.Module):
    """
    x_bg: (B,L,N,Fbg)
    Layout in your data (block-major):
      dyn = [an block (K points each has 4 vars), fc1 block, ..., fcH block]
      coords tail = [dx_k1..kK, dy_k1..kK, dist_k1..kK] => 3K

    Steps:
      1) decode dyn -> (B,L,N,K,Fp)
      2) feat_proj + pos_mlp -> p (B,L,N,K,D)
      3) OPTIONAL: local SetConv over K points (captures neighbor interactions)
      4) attention pool over K -> z (B,L,N,D)
    """
    def __init__(
        self,
        d_model,
        K=16,
        H=6,
        feat_per_point=4,
        use_forecast=True,
        use_setconv: bool = True,
    ):
        super().__init__()
        self.K = int(K)
        self.H = int(H)
        self.feat_per_point = int(feat_per_point)
        self.use_forecast = bool(use_forecast)

        self.num_blocks = 1 + (self.H if self.use_forecast else 0)  # an + fc1..H
        self.Fp = self.feat_per_point * self.num_blocks

        self.use_setconv = bool(use_setconv)

        self.feat_proj = nn.Linear(self.Fp, d_model)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        if self.use_setconv:
            self.k_setconv = BgPointSetConv(d_model=d_model)

        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self.dist_scale = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_bg, x_bg_valid=None):
        B, L, N, Fbg = x_bg.shape
        K = self.K
        coord_dim = 3 * K
        if Fbg <= coord_dim:
            raise RuntimeError(f"BgGridEncoder expects coords last {coord_dim} dims, but Fbg={Fbg}")

        dyn = x_bg[..., :Fbg - coord_dim]
        crd = x_bg[..., Fbg - coord_dim:]

        dx = crd[..., 0:K]
        dy = crd[..., K:2*K]
        ds = crd[..., 2*K:3*K]
        pos = torch.stack([dx, dy, ds], dim=-1)   # (B,L,N,K,3)

        expected_dyn = self.num_blocks * K * self.feat_per_point
        if dyn.shape[-1] != expected_dyn:
            raise RuntimeError(
                f"BgGridEncoder dyn dim mismatch: got {dyn.shape[-1]} expect {expected_dyn} "
                f"(blocks={self.num_blocks}, K={K}, feat_per_point={self.feat_per_point})"
            )

        # decode block-major: (blocks,K,4) -> (K,blocks,4) -> (K,Fp)
        dyn = dyn.view(B, L, N, self.num_blocks, K, self.feat_per_point)
        dyn = dyn.permute(0,1,2,4,3,5).contiguous()     # (B,L,N,K,blocks,4)
        dyn = dyn.view(B, L, N, K, self.Fp)             # (B,L,N,K,Fp)

        p = self.feat_proj(dyn)                         # (B,L,N,K,D)
        p = p + self.pos_mlp(pos)                       # inject dx/dy/dist

        # valid mask for points
        if x_bg_valid is not None:
            vdyn = x_bg_valid[..., :Fbg - coord_dim]
            vdyn = vdyn.view(B, L, N, self.num_blocks, K, self.feat_per_point)
            vdyn = vdyn.permute(0,1,2,4,3,5).contiguous()     # (B,L,N,K,blocks,4)
            v_point = (vdyn.mean(dim=(-1,-2)) > 0.5).float()  # (B,L,N,K)
        else:
            v_point = torch.ones((B, L, N, K), device=x_bg.device, dtype=p.dtype)

        # ===== NEW: local SetConv inside K =====
        if self.use_setconv:
            p = self.k_setconv(p, pos, v=v_point)       # (B,L,N,K,D)

        a = self.attn(p).squeeze(-1)                    # (B,L,N,K)

        scale = F.softplus(self.dist_scale)
        a = a - scale * ds                              # ds: (B,L,N,K)

        a = a.masked_fill(v_point < 0.5, -1e9)
        w = torch.softmax(a, dim=-1).unsqueeze(-1)      # (B,L,N,K,1)

        z = (w * p).sum(dim=3)                          # (B,L,N,D)
        return self.norm(z)

# -------------------------
# DAM Soft
# -------------------------
class DAMSoft(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.shared = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(3*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.fuse = nn.Linear(2*d_model, d_model)
        self.smooth = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, z_bg, z_obs):
        y_dot = 0.5 * (self.shared(z_bg) + self.shared(z_obs))
        g_in = torch.cat([z_obs, z_bg, y_dot], dim=-1)
        alpha = torch.sigmoid(self.gate(g_in))  # (B,L,N,1)

        y_sel = alpha * z_obs + (1.0 - alpha) * z_bg
        z = self.fuse(torch.cat([y_dot, y_sel], dim=-1))

        B, L, N, D = z.shape
        z2 = z.permute(0,2,3,1).reshape(B*N, D, L)
        z2 = self.smooth(z2).reshape(B, N, D, L).permute(0,3,1,2)
        return z + z2

# -------------------------
# Full FNP Fusion Frontend
# -------------------------
class FNPFusion(nn.Module):
    """
    Route A:
      obs: FuncRepVFR (SCADA 7 dims)
      bg : BgGridEncoder (K points + coords)  [NOW with optional local SetConv]
      fuse via DAMSoft
    """
    def __init__(
        self,
        obs_dim,
        bg_dim,
        d_model=128,
        scheme="A",
        modes=8,
        K_bg=16,
        H=6,
        bg_use_setconv: bool = True,
    ):
        super().__init__()
        self.scheme = scheme.upper()
        if self.scheme != "A":
            raise RuntimeError("This version is for route A only (scheme='A').")

        self.obs_enc = FuncRepVFR(obs_dim, d_model, modes=modes)
        self.bg_enc  = BgGridEncoder(
            d_model=d_model,
            K=K_bg,
            H=H,
            feat_per_point=4,
            use_forecast=True,
            use_setconv=bg_use_setconv,
        )
        self.dam = DAMSoft(d_model)

    def forward(self, x_obs, x_bg, coords, x_bg_valid=None):
        z_obs = self.obs_enc(x_obs, coords)                    # (B,L,N,D)
        z_bg  = self.bg_enc(x_bg, x_bg_valid=x_bg_valid)       # (B,L,N,D)
        z = self.dam(z_bg, z_obs)
        return z
