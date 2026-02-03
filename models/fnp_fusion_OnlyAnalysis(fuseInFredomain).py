# models/fnp_fusion_OnlyAnalysis.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _wind_uv_from_speed_dir_sincos(speed: torch.Tensor, dir_sin: torch.Tensor, dir_cos: torch.Tensor):
    u = -speed * dir_sin
    v = -speed * dir_cos
    return u, v


def _rotate_uv_to_nacelle_frame(u: torch.Tensor, v: torch.Tensor, nac_sin: torch.Tensor, nac_cos: torch.Tensor):
    forward = u * nac_sin + v * nac_cos
    left = -u * nac_cos + v * nac_sin
    return forward, left


class SetConv(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.proj = nn.Linear(din, dout)
        self.log_lt = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, coords):
        B, P, _ = x.shape
        h = self.proj(x)

        t = coords[:, 0:1]
        xy = coords[:, 1:3]

        dt = t - t.t()
        dxy = xy[:, None, :] - xy[None, :, :]

        lt = torch.exp(self.log_lt) + 1e-6
        ls = torch.exp(self.log_ls) + 1e-6

        dist2 = (dt / lt) ** 2 + (dxy[..., 0] / ls) ** 2 + (dxy[..., 1] / ls) ** 2
        K = torch.exp(-0.5 * dist2).to(x.device)
        W = K / (K.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.einsum("pq,bqd->bpd", W, h)
        return out


class NeuralFourierLayer(nn.Module):
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

        y2 = x.permute(0, 1, 3, 2).reshape(B * N, D, L)
        y2 = self.conv(y2).reshape(B, N, D, L).permute(0, 1, 3, 2)

        out = self.norm(y + y2)
        return out


class FuncRepVFR(nn.Module):
    def __init__(self, in_dim, d_model, modes=8):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.setconv = SetConv(d_model, d_model)
        self.nfl = NeuralFourierLayer(d_model, modes=modes)

    def forward(self, x, coords):
        B, L, N, _ = x.shape
        h = self.in_proj(x)
        h = h.reshape(B, L * N, -1)
        h = self.setconv(h, coords)
        h = h.reshape(B, L, N, -1).permute(0, 2, 1, 3)
        h = self.nfl(h)
        h = h.permute(0, 2, 1, 3)
        return h


class BgPointSetConv(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.log_ls_xy = nn.Parameter(torch.tensor(0.0))
        self.log_ls_d = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, p: torch.Tensor, pos: torch.Tensor, v: torch.Tensor = None) -> torch.Tensor:
        if p.ndim != 5 or pos.ndim != 5:
            raise RuntimeError(f"BgPointSetConv expects p/pos with shape (B,*,N,K,*), got p={tuple(p.shape)}, pos={tuple(pos.shape)}")
        B, T, N, K, D = p.shape
        if D != self.d_model:
            raise RuntimeError(f"d_model mismatch: p.D={D}, expected {self.d_model}")
        if pos.shape[-1] != 3:
            raise RuntimeError(f"pos last dim must be 3 (dx,dy,dist), got {pos.shape[-1]}")

        dp = pos.unsqueeze(-2) - pos.unsqueeze(-3)
        ddx = dp[..., 0]
        ddy = dp[..., 1]
        dds = dp[..., 2]

        ls_xy = torch.exp(self.log_ls_xy) + 1e-6
        ls_d = torch.exp(self.log_ls_d) + 1e-6

        dist2 = (ddx / ls_xy) ** 2 + (ddy / ls_xy) ** 2 + (dds / ls_d) ** 2
        Kmat = torch.exp(-0.5 * dist2)

        if v is not None:
            v_src = v.unsqueeze(-2)
            Kmat = Kmat * v_src

        W = Kmat / (Kmat.sum(dim=-1, keepdim=True) + 1e-6)
        mixed = torch.einsum("btnkj,btnjd->btnkd", W, p)
        out = self.norm(p + mixed)
        return out


class GridEncoder(nn.Module):
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
            pos = pos.unsqueeze(0).unsqueeze(0)
        elif pos.ndim == 4:
            pos = pos.unsqueeze(1)
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

        pos5 = self._broadcast_pos(pos, B=B, T=T, N=N, K=self.K, device=x.device)

        p = self.feat_proj(dyn)
        p = p + self.pos_mlp(pos5)

        if valid is not None:
            if valid.ndim == 4:
                v = valid.view(B, T, N, self.K, self.feat_per_point)
            else:
                v = valid
            v_point = (v.mean(dim=-1) > 0.5).float()
        else:
            v_point = torch.ones((B, T, N, self.K), device=x.device, dtype=p.dtype)

        all_invalid = (v_point.sum(dim=-1, keepdim=True) < 0.5)
        if all_invalid.any():
            v_point = v_point.clone()
            mask0 = all_invalid.squeeze(-1)
            v_point[..., 0] = torch.where(mask0, torch.ones_like(v_point[..., 0]), v_point[..., 0])

        if self.use_setconv:
            p = self.k_setconv(p, pos5, v=v_point)

        a = self.attn(p).squeeze(-1)
        ds = pos5[..., 2]
        scale = F.softplus(self.dist_scale)
        a = a - scale * ds

        a = a.masked_fill(v_point < 0.5, -1e9)
        w = torch.softmax(a, dim=-1).unsqueeze(-1)
        z = (w * p).sum(dim=3)
        return self.norm(z)


class HorizonPool(nn.Module):
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
        a = self.score(e).squeeze(-1)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        c = (w * e).sum(dim=1)
        z = self.proj(c)
        return self.norm(z)


class TriDAMFreq(nn.Module):
    """
    Frequency-domain fusion (傅立叶变换后在频域融合):
      inputs:  z_obs, z_bg, z_fc  (B,L,N,D) real
      output:  (B,L,N,D) real

    Steps:
      1) rFFT along time L -> (B,N,F,D) complex, F=floor(L/2)+1
      2) compute per-frequency gates w_obs,w_bg,w_fc -> (B,N,F,3)
      3) fuse in frequency domain
      4) irFFT back to time domain
      5) optional temporal smoothing conv + residual + LayerNorm
    """
    def __init__(self, d_model: int, gate_hidden: int = None):
        super().__init__()
        self.d_model = int(d_model)
        h = int(gate_hidden) if gate_hidden is not None else max(32, self.d_model // 4)

        # gate on magnitudes (real features)
        self.gate = nn.Sequential(
            nn.Linear(4, h),
            nn.ReLU(),
            nn.Linear(h, 3),
        )

        # blend between mean-spectrum and gated-spectrum for stability
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid -> ~0.5 at start

        self.smooth = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, z_obs: torch.Tensor, z_bg: torch.Tensor, z_fc: torch.Tensor) -> torch.Tensor:
        if z_obs.shape != z_bg.shape or z_obs.shape != z_fc.shape:
            raise RuntimeError(f"TriDAMFreq expects same shapes, got obs={tuple(z_obs.shape)} bg={tuple(z_bg.shape)} fc={tuple(z_fc.shape)}")
        if z_obs.ndim != 4:
            raise RuntimeError(f"TriDAMFreq expects (B,L,N,D), got {tuple(z_obs.shape)}")

        B, L, N, D = z_obs.shape
        if D != self.d_model:
            raise RuntimeError(f"d_model mismatch: got D={D}, expected {self.d_model}")

        # (B,N,L,D)
        o = z_obs.permute(0, 2, 1, 3)
        b = z_bg.permute(0, 2, 1, 3)
        f = z_fc.permute(0, 2, 1, 3)

        # rFFT along time -> (B,N,F,D) complex
        o_f = torch.fft.rfft(o, dim=2)
        b_f = torch.fft.rfft(b, dim=2)
        f_f = torch.fft.rfft(f, dim=2)

        # magnitudes for gating: (B,N,F)
        mag_o = torch.log1p(o_f.abs().mean(dim=-1))
        mag_b = torch.log1p(b_f.abs().mean(dim=-1))
        mag_f = torch.log1p(f_f.abs().mean(dim=-1))
        mag_m = (mag_o + mag_b + mag_f) / 3.0

        gate_in = torch.stack([mag_o, mag_b, mag_f, mag_m], dim=-1)  # (B,N,F,4)
        logits = self.gate(gate_in)                                  # (B,N,F,3)
        w = torch.softmax(logits, dim=-1)                            # (B,N,F,3)

        # IMPORTANT FIX:
        # keep w0/w1/w2 as (B,N,F,1) to broadcast to (B,N,F,D)
        w0 = w[..., 0:1]
        w1 = w[..., 1:2]
        w2 = w[..., 2:3]

        sel_f = w0 * o_f + w1 * b_f + w2 * f_f                      # (B,N,F,D) complex
        mean_f = (o_f + b_f + f_f) / 3.0

        a = torch.sigmoid(self.alpha)                                # scalar
        y_f = (1.0 - a) * mean_f + a * sel_f                         # (B,N,F,D) complex

        # back to time: (B,N,L,D) real
        y = torch.fft.irfft(y_f, n=L, dim=2)

        # (B,L,N,D)
        z = y.permute(0, 2, 1, 3)

        # temporal smoothing conv
        z2 = z.permute(0, 2, 3, 1).reshape(B * N, D, L)
        z2 = self.smooth(z2).reshape(B, N, D, L).permute(0, 3, 1, 2)

        out = self.norm(z + z2)
        return out


class VecFiLM(nn.Module):
    def __init__(self, d_model: int, dropout_p: float = 0.05):
        super().__init__()
        self.dropout = nn.Dropout(p=float(dropout_p))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, z_vec: torch.Tensor):
        z = self.dropout(z_vec)
        gb = self.mlp(z)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        return gamma, beta


class FNPFusion(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        modes: int = 8,
        K_bg: int = 16,
        bg_use_setconv: bool = True,
        fc_use_setconv: bool = True,
        fc_emb_scalar: int = 32,
        fc_emb_vec: int = 32,
        film_dropout_p: float = 0.05,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(K_bg)

        self.fc_emb_scalar = int(fc_emb_scalar)
        self.fc_emb_vec = int(fc_emb_vec)

        # scalar: [P3, dP3, W3]
        self.obs_scalar_enc = FuncRepVFR(in_dim=3, d_model=self.d_model, modes=modes)

        self.bg_scalar_enc = GridEncoder(
            d_model=self.d_model,
            K=self.K,
            feat_per_point=1,
            use_setconv=bool(bg_use_setconv),
        )

        self.fc_scalar_enc = GridEncoder(
            d_model=self.fc_emb_scalar,
            K=self.K,
            feat_per_point=1,
            use_setconv=bool(fc_use_setconv),
        )
        self.fc_scalar_pool = HorizonPool(e_dim=self.fc_emb_scalar, d_model=self.d_model)

        # === frequency-domain fusion for scalar branch ===
        self.tri_scalar = TriDAMFreq(self.d_model)

        # vector branch (nacelle frame)
        self.obs_vec_enc = FuncRepVFR(in_dim=2, d_model=self.d_model, modes=modes)

        self.bg_vec_enc = GridEncoder(
            d_model=self.d_model,
            K=self.K,
            feat_per_point=2,
            use_setconv=bool(bg_use_setconv),
        )

        self.fc_vec_enc = GridEncoder(
            d_model=self.fc_emb_vec,
            K=self.K,
            feat_per_point=2,
            use_setconv=bool(fc_use_setconv),
        )
        self.fc_vec_pool = HorizonPool(e_dim=self.fc_emb_vec, d_model=self.d_model)

        # === frequency-domain fusion for vector branch ===
        self.tri_vec = TriDAMFreq(self.d_model)

        # ONLY FiLM
        self.vec_film = VecFiLM(self.d_model, dropout_p=film_dropout_p)
        self.film_scale = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ~ 0.12
        self.out_norm = nn.LayerNorm(self.d_model)

    def forward(self, x_obs, x_an, coords, pos, fc0, x_an_valid=None, fc0v=None):
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

        dyn_an = x_an.view(B, L, N, self.K, 4)
        dyn_an_v = x_an_valid.view(B, L, N, self.K, 4) if x_an_valid is not None else None

        # ===== scalar branch =====
        x_obs_scalar = x_obs[..., 0:3]  # [P3,dP3,W3]

        an_speed = dyn_an[..., 0:1]
        an_speed_v = dyn_an_v[..., 0:1] if dyn_an_v is not None else None

        fc0_speed = fc0[..., 0:1]
        fc0_speed_v = fc0v[..., 0:1] if fc0v is not None else None

        z_obs_s = self.obs_scalar_enc(x_obs_scalar, coords)                 # (B,L,N,D)
        z_bg_s = self.bg_scalar_enc(an_speed, pos, valid=an_speed_v)        # (B,L,N,D)

        e_fc_s = self.fc_scalar_enc(fc0_speed, pos, valid=fc0_speed_v)      # (B,H,N,Es)
        z_fc_s0 = self.fc_scalar_pool(e_fc_s)                               # (B,N,D)
        z_fc_s = z_fc_s0.unsqueeze(1).expand(B, L, N, self.d_model)          # (B,L,N,D)

        z_scalar = self.tri_scalar(z_obs=z_obs_s, z_bg=z_bg_s, z_fc=z_fc_s) # (B,L,N,D)

        # ===== vector branch (nacelle frame) =====
        W3 = x_obs[..., 2]
        dir_sin = x_obs[..., 3]
        dir_cos = x_obs[..., 4]
        nac_sin = x_obs[..., 5]
        nac_cos = x_obs[..., 6]

        u_sc, v_sc = _wind_uv_from_speed_dir_sincos(W3, dir_sin, dir_cos)
        f_sc, l_sc = _rotate_uv_to_nacelle_frame(u_sc, v_sc, nac_sin, nac_cos)
        x_obs_vec = torch.stack([f_sc, l_sc], dim=-1)                       # (B,L,N,2)

        an_uv = dyn_an[..., 2:4]                                            # (B,L,N,K,2) [u,v]
        u_an = an_uv[..., 0]
        v_an = an_uv[..., 1]
        nac_sin_k = nac_sin.unsqueeze(-1)
        nac_cos_k = nac_cos.unsqueeze(-1)
        f_an, l_an = _rotate_uv_to_nacelle_frame(u_an, v_an, nac_sin_k, nac_cos_k)
        an_vec = torch.stack([f_an, l_an], dim=-1)                          # (B,L,N,K,2)
        an_uv_v = dyn_an_v[..., 2:4] if dyn_an_v is not None else None

        fc_uv = fc0[..., 2:4]                                               # (B,H,N,K,2)
        u_fc = fc_uv[..., 0]
        v_fc = fc_uv[..., 1]
        nac_sin_t0 = nac_sin[:, -1].unsqueeze(1).unsqueeze(-1)
        nac_cos_t0 = nac_cos[:, -1].unsqueeze(1).unsqueeze(-1)
        f_fc, l_fc = _rotate_uv_to_nacelle_frame(u_fc, v_fc, nac_sin_t0, nac_cos_t0)
        fc_vec = torch.stack([f_fc, l_fc], dim=-1)                          # (B,H,N,K,2)
        fc_uv_v = fc0v[..., 2:4] if fc0v is not None else None

        z_obs_v = self.obs_vec_enc(x_obs_vec, coords)                       # (B,L,N,D)
        z_bg_v = self.bg_vec_enc(an_vec, pos, valid=an_uv_v)                # (B,L,N,D)

        e_fc_v = self.fc_vec_enc(fc_vec, pos, valid=fc_uv_v)                # (B,H,N,Ev)
        z_fc_v0 = self.fc_vec_pool(e_fc_v)                                  # (B,N,D)
        z_fc_v = z_fc_v0.unsqueeze(1).expand(B, L, N, self.d_model)          # (B,L,N,D)

        z_vec = self.tri_vec(z_obs=z_obs_v, z_bg=z_bg_v, z_fc=z_fc_v)        # (B,L,N,D)

        # ===== FiLM modulation =====
        gamma, beta = self.vec_film(z_vec)
        s_film = torch.sigmoid(self.film_scale)
        z = (1.0 + s_film * gamma) * z_scalar + (s_film * beta)
        z = self.out_norm(z)

        e_fc = torch.cat([e_fc_s, e_fc_v], dim=-1)                          # (B,H,N,Es+Ev)
        return z, e_fc
