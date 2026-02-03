# models/fnp_fusion_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# SetConv (turbine-level spatiotemporal mixing)
# -------------------------
class SetConv(nn.Module):
    """
    coords: (P, 3) where P=L*N, 3=(t,x,y)
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
# Neural Fourier Layer (time)
# -------------------------
class NeuralFourierLayer(nn.Module):
    def __init__(self, d_model, modes=8, kernel_size=3):
        super().__init__()
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

        return self.norm(x + y + y2)

# -------------------------
# Obs encoder: embed -> SetConv -> NFL
# -------------------------
class FuncRepObs(nn.Module):
    """
    x: (B,L,N,7)
    out: (B,L,N,D)
    """
    def __init__(self, in_dim, d_model, modes=8, nfl_layers=2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.setconv = SetConv(d_model, d_model)
        self.nfl = nn.ModuleList([NeuralFourierLayer(d_model, modes=modes) for _ in range(nfl_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, coords):
        B, L, N, _ = x.shape
        h0 = self.in_proj(x)                          # (B,L,N,D)

        h = h0.reshape(B, L*N, -1)
        h = self.setconv(h, coords)                   # (B,P,D)
        h = h.reshape(B, L, N, -1)                    # (B,L,N,D)

        ht = h.permute(0,2,1,3).contiguous()          # (B,N,L,D)
        for layer in self.nfl:
            ht = layer(ht)
        h = ht.permute(0,2,1,3).contiguous()          # (B,L,N,D)

        return self.norm(h + h0)                      # residual

# -------------------------
# helpers: broadcast dx/dy/ds
# -------------------------
def _as_5d_pos(t: torch.Tensor, B: int, L: int, N: int, K: int) -> torch.Tensor:
    """
    Normalize position-like tensor to shape (B,L,N,K,1) by explicit expand (no implicit broadcast in cat).
    Accepts common shapes:
      (1,1,N,K,1), (1,L,N,K,1), (B,1,N,K,1), (B,L,N,K,1)
      (N,K) / (N,K,1) / (1,N,K) / (1,N,K,1) etc.
    """
    if t is None:
        raise RuntimeError("pos tensor is None")

    # move to 5D
    if t.dim() == 2:          # (N,K)
        t = t.unsqueeze(0).unsqueeze(0).unsqueeze(-1)     # (1,1,N,K,1)
    elif t.dim() == 3:
        # could be (N,K,1) or (1,N,K) etc.
        if t.shape[-1] == 1 and t.shape[0] == N and t.shape[1] == K:
            t = t.unsqueeze(0).unsqueeze(0)               # (1,1,N,K,1)
        elif t.shape[0] == 1 and t.shape[1] == N and t.shape[2] == K:
            t = t.unsqueeze(0).unsqueeze(-1)              # (1,1,N,K,1)
        else:
            # fallback: treat as (1,N,K) -> (1,1,N,K,1)
            t = t.reshape(1, 1, N, K, 1)
    elif t.dim() == 4:
        # likely (1,N,K,1) -> (1,1,N,K,1) or (B,N,K,1) -> (B,1,N,K,1)
        if t.shape[-1] != 1:
            t = t.unsqueeze(-1)
        if t.shape[0] == B and t.shape[1] == N and t.shape[2] == K:
            t = t.unsqueeze(1)                            # (B,1,N,K,1)
        elif t.shape[0] == 1 and t.shape[1] == N and t.shape[2] == K:
            t = t.unsqueeze(1)                            # (1,1,N,K,1)
        else:
            t = t.reshape(1, 1, N, K, 1)
    elif t.dim() != 5:
        raise RuntimeError(f"Unsupported pos tensor dim={t.dim()} shape={tuple(t.shape)}")

    # explicit expand to (B,L,N,K,1)
    if t.shape[-1] != 1:
        t = t[..., :1]

    if t.shape[2] != N or t.shape[3] != K:
        raise RuntimeError(f"pos tensor N/K mismatch: got {tuple(t.shape)} expect (*,*,{N},{K},1)")

    t = t.expand(B, L, N, K, 1)
    return t

# -------------------------
# BG encoder (analysis-only for history)
# -------------------------
class BgHistEncoder(nn.Module):
    """
    x_bg_an: (B,L,N,K,V)
    dx/dy/ds: any broadcastable common shape -> normalized to (B,L,N,K,1)
    out: (B,L,N,D)
    """
    def __init__(self, d_model, vars_per_point=2, use_dist_bias=True):
        super().__init__()
        self.vars_per_point = int(vars_per_point)
        self.use_dist_bias = bool(use_dist_bias)

        in_dim = self.vars_per_point + 3  # (point vars) + (dx,dy,dist)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.dist_scale = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_bg_an, dx, dy, ds, v_point=None):
        B, L, N, K, V = x_bg_an.shape

        dx = _as_5d_pos(dx, B=B, L=L, N=N, K=K)
        dy = _as_5d_pos(dy, B=B, L=L, N=N, K=K)
        ds = _as_5d_pos(ds, B=B, L=L, N=N, K=K)

        feat = torch.cat([x_bg_an, dx, dy, ds], dim=-1)     # (B,L,N,K,V+3)
        p = self.point_mlp(feat)                            # (B,L,N,K,D)
        a = self.attn(p).squeeze(-1)                        # (B,L,N,K)

        if self.use_dist_bias:
            a = a - F.softplus(self.dist_scale) * ds.squeeze(-1)

        if v_point is not None:
            a = a.masked_fill(v_point < 0.5, -1e9)

        w = torch.softmax(a, dim=-1).unsqueeze(-1)          # (B,L,N,K,1)
        z = (w * p).sum(dim=3)                              # (B,L,N,D)
        return self.norm(z)

class BgFutureEncoder(nn.Module):
    """
    x_bg_fc0: (B,H,N,K,V)
    out: (B,H,N,D)
    """
    def __init__(self, d_model, vars_per_point=2, use_dist_bias=True):
        super().__init__()
        self.enc = BgHistEncoder(d_model=d_model, vars_per_point=vars_per_point, use_dist_bias=use_dist_bias)

    def forward(self, x_bg_fc0, dx, dy, ds, v_point=None):
        # Here L == H (we reuse BgHistEncoder semantics: treat H as the "time" dim)
        return self.enc(x_bg_fc0, dx, dy, ds, v_point=v_point)

# -------------------------
# DAM hard + smoothing (paper-style)
# -------------------------
class DAMHard(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)
        self.fuse = nn.Linear(2*d_model, d_model)
        self.smooth = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, z_bg, z_obs):
        shared = 0.5 * (self.lin(z_bg) + self.lin(z_obs))  # (B,L,N,D)

        e_bg  = ((z_bg  - shared)**2).mean(dim=-1, keepdim=True)
        e_obs = ((z_obs - shared)**2).mean(dim=-1, keepdim=True)

        sel = torch.where(e_bg <= e_obs, z_bg, z_obs)
        h = self.fuse(torch.cat([shared, sel], dim=-1))     # (B,L,N,D)

        B, L, N, D = h.shape
        ht = h.permute(0,2,3,1).reshape(B*N, D, L)
        ht = self.smooth(ht).reshape(B, N, D, L).permute(0,3,1,2)
        return h + ht

# -------------------------
# Full FNP Fusion v3
# -------------------------
class FNPFusionV3(nn.Module):
    """
    Input:
      x_obs: (B,L,N,7)
      x_bg : (B,L,N,bg_dim) where bg_dim = blocks*(K*V)
    Output:
      z_hist: (B,L,N,D)
      z_bg_future: (B,H,N,D)
    """
    def __init__(self, d_model=128, modes=8, nfl_layers=2, K=16, H=6, vars_per_point=2):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(K)
        self.H = int(H)
        self.V = int(vars_per_point)

        self.an_dim = self.K * self.V
        self.num_blocks = 1 + self.H  # an + fc1..H

        self.obs_enc = FuncRepObs(in_dim=7, d_model=d_model, modes=modes, nfl_layers=nfl_layers)
        self.bg_hist_enc = BgHistEncoder(d_model=d_model, vars_per_point=self.V, use_dist_bias=True)
        self.bg_fut_enc  = BgFutureEncoder(d_model=d_model, vars_per_point=self.V, use_dist_bias=True)
        self.dam = DAMHard(d_model=d_model)

    def forward(self, x_obs, x_bg, coords, dx, dy, ds, x_obs_v=None, x_bg_v=None):
        B, L, N, _ = x_obs.shape
        K, H, V = self.K, self.H, self.V

        # ---- obs ----
        z_obs = self.obs_enc(x_obs, coords)  # (B,L,N,D)

        # ---- bg history: ONLY analysis block ----
        bg_an = x_bg[..., :self.an_dim]  # (B,L,N,an_dim)
        bg_an = bg_an.view(B, L, N, K, V)

        if x_bg_v is not None:
            v_an = x_bg_v[..., :self.an_dim].view(B, L, N, K, V)
            v_point = (v_an.mean(dim=-1) > 0.5).float()  # (B,L,N,K)
        else:
            v_point = None

        # dx/dy/ds may be (1,1,N,K,1); BgHistEncoder will expand explicitly
        z_bg = self.bg_hist_enc(bg_an, dx, dy, ds, v_point=v_point)  # (B,L,N,D)

        # ---- fuse ----
        z_hist = self.dam(z_bg, z_obs)  # (B,L,N,D)

        # ---- bg future from t0 forecast ONLY ----
        x_last_bg = x_bg[:, -1]  # (B,N,bg_dim)
        if x_bg_v is not None:
            x_last_bg_v = x_bg_v[:, -1]
        else:
            x_last_bg_v = None

        fc_blocks = []
        fc_v_blocks = []
        for lead in range(1, H + 1):
            s = self.an_dim + (lead - 1) * self.an_dim
            e = s + self.an_dim
            blk = x_last_bg[:, :, s:e].view(B, N, K, V)
            fc_blocks.append(blk)
            if x_last_bg_v is not None:
                vb = x_last_bg_v[:, :, s:e].view(B, N, K, V)
                fc_v_blocks.append(vb)

        x_fc0 = torch.stack(fc_blocks, dim=1)  # (B,H,N,K,V)
        if fc_v_blocks:
            v_fc0 = torch.stack(fc_v_blocks, dim=1)                # (B,H,N,K,V)
            v_fc0 = (v_fc0.mean(dim=-1) > 0.5).float()             # (B,H,N,K)
        else:
            v_fc0 = None

        # dx/dy/ds can still be (1,1,N,K,1); BgFutureEncoder will expand
        z_bg_future = self.bg_fut_enc(x_fc0, dx, dy, ds, v_point=v_fc0)  # (B,H,N,D)
        return z_hist, z_bg_future
