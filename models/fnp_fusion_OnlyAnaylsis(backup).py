# models/fnp_fusion_OnlyAnalysis.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Helpers: vector construction + rotate to nacelle reference
# =========================================================

def _wind_uv_from_speed_dir_sincos(speed: torch.Tensor, dir_sin: torch.Tensor, dir_cos: torch.Tensor):
    """
    Convert wind speed + wind-from direction (sin/cos) to (u,v) wind vector
    in meteorological convention:
      u: eastward (m/s), v: northward (m/s)
    direction is wind-from, so the blowing-to vector is:
      u = -speed * sin(dir)
      v = -speed * cos(dir)
    Shapes:
      speed, dir_sin, dir_cos: (...,)
    Returns:
      u, v: (...,)
    """
    u = -speed * dir_sin
    v = -speed * dir_cos
    return u, v


def _rotate_uv_to_nacelle_frame(u: torch.Tensor, v: torch.Tensor, nac_sin: torch.Tensor, nac_cos: torch.Tensor):
    """
    Rotate global (u,v) (east,north) into nacelle reference frame.

    Here nac angle is absolute azimuth measured from North clockwise (same as SCADA columns).
    Define nacelle forward unit vector in (E,N) as:
      e_forward = (sin(nac), cos(nac))
    Define nacelle left unit vector as:
      e_left = (-cos(nac), sin(nac))  (90 deg CCW from forward)

    Then:
      forward = dot([u,v], e_forward) = u*sin(nac) + v*cos(nac)
      left    = dot([u,v], e_left)    = -u*cos(nac) + v*sin(nac)

    Shapes broadcast naturally.
    Returns:
      forward, left  (same shape as u/v)
    """
    forward = u * nac_sin + v * nac_cos
    left = -u * nac_cos + v * nac_sin
    return forward, left


# -------------------------
# SetConv (kernel-based set convolution)
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
        Kf = x_fft.size(2)

        Wr = self.Wr[:Kf]
        Wi = self.Wi[:Kf]

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
# Scheme for (turbine history) sequence -> functional rep
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
# local SetConv over K neighbor grid points (per (B,L,N) or (B,H,N))
# -------------------------
class BgPointSetConv(nn.Module):
    """
    p:   (B,*,N,K,D)
    pos: (B,*,N,K,3) where 3=(dx,dy,dist) normalized
    v:   (B,*,N,K)   point valid mask (0/1)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.log_ls_xy = nn.Parameter(torch.tensor(0.0))
        self.log_ls_d  = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, p: torch.Tensor, pos: torch.Tensor, v: torch.Tensor = None) -> torch.Tensor:
        if p.ndim != 5 or pos.ndim != 5:
            raise RuntimeError(f"BgPointSetConv expects p/pos with shape (B,*,N,K,*), got p={tuple(p.shape)}, pos={tuple(pos.shape)}")
        B, T, N, K, D = p.shape
        if D != self.d_model:
            raise RuntimeError(f"d_model mismatch: p.D={D}, expected {self.d_model}")
        if pos.shape[-1] != 3:
            raise RuntimeError(f"pos last dim must be 3 (dx,dy,dist), got {pos.shape[-1]}")

        dp = pos.unsqueeze(-2) - pos.unsqueeze(-3)  # (B,T,N,K,K,3)
        ddx = dp[..., 0]
        ddy = dp[..., 1]
        dds = dp[..., 2]

        ls_xy = torch.exp(self.log_ls_xy) + 1e-6
        ls_d  = torch.exp(self.log_ls_d)  + 1e-6

        dist2 = (ddx / ls_xy) ** 2 + (ddy / ls_xy) ** 2 + (dds / ls_d) ** 2
        Kmat = torch.exp(-0.5 * dist2)  # (B,T,N,K,K)

        if v is not None:
            v_src = v.unsqueeze(-2)      # (B,T,N,1,K)
            Kmat = Kmat * v_src

        W = Kmat / (Kmat.sum(dim=-1, keepdim=True) + 1e-6)
        mixed = torch.einsum("btnkj,btnjd->btnkd", W, p)
        out = self.norm(p + mixed)
        return out


# -------------------------
# Generic grid encoder (history/forecast), supports scalar or vector per-point feats
# -------------------------
class GridEncoder(nn.Module):
    """
    x: (B,T,N,K,feat_per_point)  OR (B,T,N,K*feat_per_point)
    pos: (N,K,3) or (B,N,K,3) or (B,T,N,K,3)
    valid: same layout as x, optional (0/1)

    output: (B,T,N,D)
    """
    def __init__(self, d_model: int, K: int = 16, feat_per_point: int = 4, use_setconv: bool = True):
        super().__init__()
        self.K = int(K)
        self.feat_per_point = int(feat_per_point)
        self.use_setconv = bool(use_setconv)
        self.d_model = int(d_model)

        self.feat_proj = nn.Linear(self.feat_per_point, self.d_model)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        if self.use_setconv:
            self.k_setconv = BgPointSetConv(d_model=self.d_model)

        self.attn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )

        self.dist_scale = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(self.d_model)

    def _broadcast_pos(self, pos: torch.Tensor, B: int, T: int, N: int, K: int, device) -> torch.Tensor:
        if pos.ndim == 3:
            pos = pos.unsqueeze(0).unsqueeze(0)  # (1,1,N,K,3)
        elif pos.ndim == 4:
            pos = pos.unsqueeze(1)               # (B,1,N,K,3)
        elif pos.ndim == 5:
            pass
        else:
            raise RuntimeError(f"pos must be (N,K,3)/(B,N,K,3)/(B,T,N,K,3), got {tuple(pos.shape)}")
        pos = pos.to(device)
        if pos.size(0) == 1 and B > 1:
            pos = pos.expand(B, -1, -1, -1, -1)
        if pos.size(1) == 1 and T > 1:
            pos = pos.expand(-1, T, -1, -1, -1)
        if pos.size(2) != N or pos.size(3) != K:
            raise RuntimeError(f"pos shape mismatch after broadcast: {tuple(pos.shape)} vs expected (B,T,N,K,3)=({B},{T},{N},{K},3)")
        return pos

    def forward(self, x: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor = None) -> torch.Tensor:
        if x.ndim == 4:
            B, T, N, Fdim = x.shape
            if Fdim != self.K * self.feat_per_point:
                raise RuntimeError(f"x last dim must be K*feat_per_point={self.K*self.feat_per_point}, got {Fdim}")
            dyn = x.view(B, T, N, self.K, self.feat_per_point)
        elif x.ndim == 5:
            B, T, N, K, fp = x.shape
            if K != self.K or fp != self.feat_per_point:
                raise RuntimeError(f"x must be (B,T,N,K,{self.feat_per_point}) with K={self.K}, got {tuple(x.shape)}")
            dyn = x
        else:
            raise RuntimeError(f"x must be (B,T,N,K*fp) or (B,T,N,K,fp), got {tuple(x.shape)}")

        pos5 = self._broadcast_pos(pos, B=B, T=T, N=N, K=self.K, device=x.device)  # (B,T,N,K,3)

        p = self.feat_proj(dyn)      # (B,T,N,K,D)
        p = p + self.pos_mlp(pos5)

        if valid is not None:
            if valid.ndim == 4:
                v = valid.view(B, T, N, self.K, self.feat_per_point)
            else:
                v = valid
            v_point = (v.mean(dim=-1) > 0.5).float()
        else:
            v_point = torch.ones((B, T, N, self.K), device=x.device, dtype=p.dtype)

        if self.use_setconv:
            p = self.k_setconv(p, pos5, v=v_point)

        a = self.attn(p).squeeze(-1)  # (B,T,N,K)
        ds = pos5[..., 2]
        scale = F.softplus(self.dist_scale)
        a = a - scale * ds

        a = a.masked_fill(v_point < 0.5, -1e9)
        w = torch.softmax(a, dim=-1).unsqueeze(-1)
        z = (w * p).sum(dim=3)  # (B,T,N,D)
        return self.norm(z)


# -------------------------
# Forecast horizon pooling (over H)
# -------------------------
class HorizonPool(nn.Module):
    """
    e: (B,H,N,E) -> c: (B,N,E) -> z: (B,N,D)
    """
    def __init__(self, e_dim: int, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(e_dim, e_dim),
            nn.ReLU(),
            nn.Linear(e_dim, 1),
        )
        self.proj = nn.Linear(e_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        if e.ndim != 4:
            raise RuntimeError(f"e must be (B,H,N,E), got {tuple(e.shape)}")
        a = self.score(e).squeeze(-1)           # (B,H,N)
        w = torch.softmax(a, dim=1).unsqueeze(-1)  # (B,H,N,1)
        c = (w * e).sum(dim=1)                  # (B,N,E)
        z = self.proj(c)                        # (B,N,D)
        return self.norm(z)


# -------------------------
# Tri-way fusion (soft gating)
# -------------------------
class TriDAMSoft(nn.Module):
    """
    Inputs: z_obs, z_bg, z_fc each (B,L,N,D)
    Output: (B,L,N,D)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.shared = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(4*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),
        )
        self.fuse = nn.Linear(2*d_model, d_model)
        self.smooth = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_obs: torch.Tensor, z_bg: torch.Tensor, z_fc: torch.Tensor) -> torch.Tensor:
        if z_obs.shape != z_bg.shape or z_obs.shape != z_fc.shape:
            raise RuntimeError(f"TriDAM expects same shapes, got obs={tuple(z_obs.shape)} bg={tuple(z_bg.shape)} fc={tuple(z_fc.shape)}")

        y_mean = (self.shared(z_obs) + self.shared(z_bg) + self.shared(z_fc)) / 3.0  # (B,L,N,D)
        g_in = torch.cat([z_obs, z_bg, z_fc, y_mean], dim=-1)                        # (B,L,N,4D)
        logits = self.gate(g_in)                                                     # (B,L,N,3)
        w = torch.softmax(logits, dim=-1)                                            # (B,L,N,3)

        y_sel = w[..., 0:1] * z_obs + w[..., 1:2] * z_bg + w[..., 2:3] * z_fc        # (B,L,N,D)
        z = self.fuse(torch.cat([y_mean, y_sel], dim=-1))                             # (B,L,N,D)

        B, L, N, D = z.shape
        z2 = z.permute(0,2,3,1).reshape(B*N, D, L)
        z2 = self.smooth(z2).reshape(B, N, D, L).permute(0,3,1,2)

        out = self.norm(z + z2)
        return out


# -------------------------
# Vector -> scalar modulation (FiLM)
# -------------------------
class VecFiLM(nn.Module):
    """
    Given vector-branch representation z_vec (B,L,N,D),
    produce (gamma, beta) to modulate scalar representation z_s (B,L,N,D):
      z = (1 + tanh(gamma)) * z_s + beta
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, z_vec: torch.Tensor):
        gb = self.mlp(z_vec)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        gamma = torch.tanh(gamma)
        return gamma, beta


# =========================================================
# Full FNP Fusion with Vector Branch (nacelle-frame)
# =========================================================
class FNPFusion(nn.Module):
    """
    Scalar branch (3-way):
      - SCADA scalar history:    [P3, dP3, W3]
      - CERRA scalar history:    speed only (K points)
      - CERRA scalar forecast:   speed only (K points) -> horizon context

    Vector branch (3-way, nacelle-frame):
      - SCADA wind vector:       (u,v) from [W3, dir_sin/cos], rotate by nac_sin/cos -> (forward,left)
      - CERRA history vector:    (u,v) from analysis, rotate by nac_sin/cos
      - CERRA forecast vector:   (u,v) from forecast@t0, rotate by nac(t0)

    Outputs:
      z   : (B,L,N,D)   fused history representation (scalar fused, modulated by vector)
      e_fc: (B,H,N,E_total) per-horizon forecast embedding for decoder exog (scalar+vector concat)
    """
    def __init__(
        self,
        d_model: int = 128,
        modes: int = 8,
        K_bg: int = 16,
        bg_use_setconv: bool = True,
        fc_use_setconv: bool = True,
        fc_emb_scalar: int = 32,
        fc_emb_vec: int = 32,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(K_bg)

        self.fc_emb_scalar = int(fc_emb_scalar)
        self.fc_emb_vec = int(fc_emb_vec)

        # ---------- scalar encoders ----------
        # SCADA scalar: [P3, dP3, W3] => 3 dims
        self.obs_scalar_enc = FuncRepVFR(in_dim=3, d_model=self.d_model, modes=modes)

        # CERRA scalar history: speed only => 1 dim
        self.bg_scalar_enc = GridEncoder(
            d_model=self.d_model,
            K=self.K,
            feat_per_point=1,
            use_setconv=bool(bg_use_setconv),
        )

        # CERRA scalar forecast: speed only => 1 dim -> e_fc_scalar (B,H,N,E_s)
        self.fc_scalar_enc = GridEncoder(
            d_model=self.fc_emb_scalar,
            K=self.K,
            feat_per_point=1,
            use_setconv=bool(fc_use_setconv),
        )
        self.fc_scalar_pool = HorizonPool(e_dim=self.fc_emb_scalar, d_model=self.d_model)

        self.tri_scalar = TriDAMSoft(self.d_model)

        # ---------- vector encoders (nacelle-frame) ----------
        # SCADA vector: (forward,left) => 2 dims
        self.obs_vec_enc = FuncRepVFR(in_dim=2, d_model=self.d_model, modes=modes)

        # CERRA vector history: (u,v)->(forward,left) => 2 dims
        self.bg_vec_enc = GridEncoder(
            d_model=self.d_model,
            K=self.K,
            feat_per_point=2,
            use_setconv=bool(bg_use_setconv),
        )

        # CERRA vector forecast: (u,v)->(forward,left) => 2 dims -> e_fc_vec (B,H,N,E_v)
        self.fc_vec_enc = GridEncoder(
            d_model=self.fc_emb_vec,
            K=self.K,
            feat_per_point=2,
            use_setconv=bool(fc_use_setconv),
        )
        self.fc_vec_pool = HorizonPool(e_dim=self.fc_emb_vec, d_model=self.d_model)

        self.tri_vec = TriDAMSoft(self.d_model)

        # ---------- vector -> scalar modulation ----------
        self.vec_film = VecFiLM(self.d_model)
        self.vec_to_scalar = nn.Linear(self.d_model, self.d_model)
        self.out_norm = nn.LayerNorm(self.d_model)

    def forward(self, x_obs, x_an, coords, pos, fc0, x_an_valid=None, fc0v=None):
        """
        x_obs: (B,L,N,7)  = [P3,dP3,W3, dir_sin,dir_cos, nac_sin,nac_cos]
        x_an : (B,L,N,K*4) analysis history, var_order=[speed,direction,u,v]
        fc0  : (B,H,N,K,4) forecast@t0, var_order=[speed,direction,u,v]
        pos  : (N,K,3) or (B,N,K,3)
        x_an_valid: same layout as x_an
        fc0v: same layout as fc0
        """
        if x_obs.ndim != 4:
            raise RuntimeError(f"x_obs must be (B,L,N,7), got {tuple(x_obs.shape)}")
        if x_an.ndim != 4:
            raise RuntimeError(f"x_an must be (B,L,N,K*4), got {tuple(x_an.shape)}")
        if fc0.ndim != 5:
            raise RuntimeError(f"fc0 must be (B,H,N,K,4), got {tuple(fc0.shape)}")

        B, L, N, Fobs = x_obs.shape
        if Fobs != 7:
            raise RuntimeError(f"x_obs last dim must be 7, got {Fobs}")
        _, H, N2, K, fp = fc0.shape
        if N2 != N or K != self.K or fp != 4:
            raise RuntimeError(f"fc0 shape mismatch, expected (B,H,N,K,4)=({B},{H},{N},{self.K},4), got {tuple(fc0.shape)}")

        # ======================
        # 1) Scalar branch inputs
        # ======================
        x_obs_scalar = x_obs[..., 0:3]  # (B,L,N,3)

        dyn_an = x_an.view(B, L, N, self.K, 4)
        an_speed = dyn_an[..., 0:1]     # (B,L,N,K,1)

        if x_an_valid is not None:
            dyn_an_v = x_an_valid.view(B, L, N, self.K, 4)
            an_speed_v = dyn_an_v[..., 0:1]
        else:
            an_speed_v = None

        fc0_speed = fc0[..., 0:1]      # (B,H,N,K,1)
        if fc0v is not None:
            fc0_speed_v = fc0v[..., 0:1]
        else:
            fc0_speed_v = None

        # encode scalar
        z_obs_s = self.obs_scalar_enc(x_obs_scalar, coords)                    # (B,L,N,D)
        z_bg_s  = self.bg_scalar_enc(an_speed, pos, valid=an_speed_v)          # (B,L,N,D)

        e_fc_s  = self.fc_scalar_enc(fc0_speed, pos, valid=fc0_speed_v)        # (B,H,N,E_s)
        z_fc_s0 = self.fc_scalar_pool(e_fc_s)                                  # (B,N,D)
        z_fc_s  = z_fc_s0.unsqueeze(1).expand(B, L, N, self.d_model)            # (B,L,N,D)

        z_scalar = self.tri_scalar(z_obs=z_obs_s, z_bg=z_bg_s, z_fc=z_fc_s)    # (B,L,N,D)

        # ======================
        # 2) Vector branch inputs (nacelle-frame)
        # ======================
        dir_sin = x_obs[..., 3]   # (B,L,N)
        dir_cos = x_obs[..., 4]
        nac_sin = x_obs[..., 5]
        nac_cos = x_obs[..., 6]
        W3 = x_obs[..., 2]        # (B,L,N) speed magnitude

        # SCADA wind vector (global u,v) -> nacelle frame (forward,left)
        u_sc, v_sc = _wind_uv_from_speed_dir_sincos(W3, dir_sin, dir_cos)      # (B,L,N)
        f_sc, l_sc = _rotate_uv_to_nacelle_frame(u_sc, v_sc, nac_sin, nac_cos) # (B,L,N)
        x_obs_vec = torch.stack([f_sc, l_sc], dim=-1)                           # (B,L,N,2)

        # CERRA analysis vector: (u,v) -> nacelle frame per time
        an_uv = dyn_an[..., 2:4]  # (B,L,N,K,2)
        u_an = an_uv[..., 0]      # (B,L,N,K)
        v_an = an_uv[..., 1]
        nac_sin_k = nac_sin.unsqueeze(-1)  # (B,L,N,1)
        nac_cos_k = nac_cos.unsqueeze(-1)
        f_an, l_an = _rotate_uv_to_nacelle_frame(u_an, v_an, nac_sin_k, nac_cos_k)  # (B,L,N,K)
        an_vec = torch.stack([f_an, l_an], dim=-1)                                   # (B,L,N,K,2)

        if x_an_valid is not None:
            an_uv_v = dyn_an_v[..., 2:4]  # (B,L,N,K,2)
        else:
            an_uv_v = None

        # CERRA forecast vector: (u,v) -> nacelle frame using nac(t0) = last history step
        fc_uv = fc0[..., 2:4]  # (B,H,N,K,2)
        u_fc = fc_uv[..., 0]   # (B,H,N,K)
        v_fc = fc_uv[..., 1]
        nac_sin_t0 = nac_sin[:, -1].unsqueeze(1).unsqueeze(-1)  # (B,1,N,1)
        nac_cos_t0 = nac_cos[:, -1].unsqueeze(1).unsqueeze(-1)
        f_fc, l_fc = _rotate_uv_to_nacelle_frame(u_fc, v_fc, nac_sin_t0, nac_cos_t0) # (B,H,N,K)
        fc_vec = torch.stack([f_fc, l_fc], dim=-1)                                     # (B,H,N,K,2)

        if fc0v is not None:
            fc_uv_v = fc0v[..., 2:4]
        else:
            fc_uv_v = None

        # encode vector
        z_obs_v = self.obs_vec_enc(x_obs_vec, coords)                         # (B,L,N,D)
        z_bg_v  = self.bg_vec_enc(an_vec, pos, valid=an_uv_v)                 # (B,L,N,D)

        e_fc_v  = self.fc_vec_enc(fc_vec, pos, valid=fc_uv_v)                 # (B,H,N,E_v)
        z_fc_v0 = self.fc_vec_pool(e_fc_v)                                    # (B,N,D)
        z_fc_v  = z_fc_v0.unsqueeze(1).expand(B, L, N, self.d_model)           # (B,L,N,D)

        z_vec = self.tri_vec(z_obs=z_obs_v, z_bg=z_bg_v, z_fc=z_fc_v)         # (B,L,N,D)

        # ======================
        # 3) Vector -> Scalar modulation (innovation point)
        # ======================
        gamma, beta = self.vec_film(z_vec)            # each (B,L,N,D)
        z = (1.0 + gamma) * z_scalar + beta
        z = self.out_norm(z + self.vec_to_scalar(z_vec))

        # decoder exog = concat scalar+vector forecast embeddings
        e_fc = torch.cat([e_fc_s, e_fc_v], dim=-1)    # (B,H,N,E_s+E_v)
        return z, e_fc
