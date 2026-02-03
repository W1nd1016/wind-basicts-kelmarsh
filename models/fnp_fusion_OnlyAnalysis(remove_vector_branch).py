# models/fnp_fusion_OnlyAnalysis_ab_remove_vec_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _nan_to_num(x: torch.Tensor, nan=0.0, posinf=0.0, neginf=0.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def _nan_to_num_complex(z: torch.Tensor) -> torch.Tensor:
    zr = _nan_to_num(z.real, nan=0.0, posinf=0.0, neginf=0.0)
    zi = _nan_to_num(z.imag, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.complex(zr, zi)


def _safe_softmax(logits: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    x = _nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    e = _nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    s = e.sum(dim=dim, keepdim=True)
    return e / (s + eps)


class SetConv(nn.Module):
    def __init__(self, din, dout, topk: int = None, attn_drop: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(din, dout)
        self.log_lt = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))
        self.topk = None if topk is None else int(topk)
        self.drop = nn.Dropout(p=float(attn_drop))

    def forward(self, x, coords):
        B, P, _ = x.shape
        x = _nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        coords = coords.to(x.device)
        coords = _nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.proj(x)

        t = coords[:, 0:1]
        xy = coords[:, 1:3]

        dt = t - t.t()
        dxy = xy[:, None, :] - xy[None, :, :]

        lt = torch.exp(self.log_lt) + 1e-6
        ls = torch.exp(self.log_ls) + 1e-6

        dist2 = (dt / lt) ** 2 + (dxy[..., 0] / ls) ** 2 + (dxy[..., 1] / ls) ** 2
        dist2 = _nan_to_num(dist2, nan=0.0, posinf=1e6, neginf=0.0)

        K = torch.exp(-0.5 * dist2)
        K = _nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

        if self.topk is not None and self.topk < P:
            topv, topi = torch.topk(K, k=self.topk, dim=-1)
            mask = torch.zeros_like(K)
            mask.scatter_(-1, topi, 1.0)
            K = K * mask

        W = K / (K.sum(dim=-1, keepdim=True) + 1e-6)
        W = self.drop(W)
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-6)
        W = _nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

        out = torch.einsum("pq,bqd->bpd", W, h)
        out = _nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class NeuralFourierLayer(nn.Module):
    def __init__(self, d_model, modes=8, kernel_size=3, spectral_drop: float = 0.0):
        super().__init__()
        self.d = d_model
        self.modes = modes
        self.spectral_drop = float(spectral_drop)

        self.Wr = nn.Parameter(torch.randn(modes, d_model, d_model) * 0.02)
        self.Wi = nn.Parameter(torch.randn(modes, d_model, d_model) * 0.02)
        self.log_gain = nn.Parameter(torch.zeros(modes))

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
        y = torch.complex(yr, yi)
        return _nan_to_num_complex(y)

    def forward(self, x):
        B, N, L, D = x.shape
        x = _nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x_fft = torch.fft.rfft(x, dim=2)
        x_fft = _nan_to_num_complex(x_fft)

        Lf = x_fft.shape[2]
        K = min(self.modes, Lf)

        y_fft = x_fft.clone()
        if K > 0:
            mix = self._complex_linear(x_fft[:, :, :K, :])
            gain = torch.exp(torch.clamp(self.log_gain[:K], -3.0, 3.0)).view(1, 1, K, 1)
            if self.training and self.spectral_drop > 0.0:
                keep = 1.0 - self.spectral_drop
                m = (torch.rand((1, 1, K, 1), device=x.device) < keep).float() / max(keep, 1e-6)
                gain = gain * m
            y_fft[:, :, :K, :] = x_fft[:, :, :K, :] + gain * mix

        y = torch.fft.irfft(y_fft, n=L, dim=2)
        y = _nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        y2 = x.permute(0, 1, 3, 2).reshape(B * N, D, L)
        y2 = self.conv(y2).reshape(B, N, D, L).permute(0, 1, 3, 2)
        y2 = _nan_to_num(y2, nan=0.0, posinf=0.0, neginf=0.0)

        out = self.norm(y + y2)
        out = _nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class FuncRepVFR(nn.Module):
    def __init__(self, in_dim, d_model, modes=8, setconv_topk: int = None, setconv_drop: float = 0.0, spectral_drop: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.setconv = SetConv(d_model, d_model, topk=setconv_topk, attn_drop=setconv_drop)
        self.nfl = NeuralFourierLayer(d_model, modes=modes, spectral_drop=spectral_drop)

    def forward(self, x, coords):
        B, L, N, _ = x.shape
        x = _nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.in_proj(x)
        h = h.reshape(B, L * N, -1)
        h = self.setconv(h, coords)
        h = h.reshape(B, L, N, -1).permute(0, 2, 1, 3)
        h = self.nfl(h)
        h = h.permute(0, 2, 1, 3)
        h = _nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
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

        p = _nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        pos = _nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

        dp = pos.unsqueeze(-2) - pos.unsqueeze(-3)
        ddx = dp[..., 0]
        ddy = dp[..., 1]
        dds = dp[..., 2]

        ls_xy = torch.exp(self.log_ls_xy) + 1e-6
        ls_d = torch.exp(self.log_ls_d) + 1e-6

        dist2 = (ddx / ls_xy) ** 2 + (ddy / ls_xy) ** 2 + (dds / ls_d) ** 2
        dist2 = _nan_to_num(dist2, nan=0.0, posinf=1e6, neginf=0.0)

        Kmat = torch.exp(-0.5 * dist2)
        Kmat = _nan_to_num(Kmat, nan=0.0, posinf=0.0, neginf=0.0)

        if v is not None:
            v = _nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            v_src = v.unsqueeze(-2)
            Kmat = Kmat * v_src

        W = Kmat / (Kmat.sum(dim=-1, keepdim=True) + 1e-6)
        W = _nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

        mixed = torch.einsum("btnkj,btnjd->btnkd", W, p)
        out = self.norm(p + mixed)
        out = _nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class GridEncoder(nn.Module):
    def __init__(self, d_model: int, K: int = 16, feat_per_point: int = 4, use_setconv: bool = True, point_drop: float = 0.0, attn_drop: float = 0.0):
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

        self.point_drop = nn.Dropout(p=float(point_drop))
        self.attn_drop = nn.Dropout(p=float(attn_drop))

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
        x = _nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

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
        pos5 = _nan_to_num(pos5, nan=0.0, posinf=0.0, neginf=0.0)

        p = self.feat_proj(dyn)
        p = p + self.pos_mlp(pos5)
        p = self.point_drop(p)
        p = _nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        if valid is not None:
            if valid.ndim == 4:
                v = valid.view(B, T, N, self.K, self.feat_per_point)
            else:
                v = valid
            v = _nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
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
        a = _nan_to_num(a, nan=-1e9, posinf=1e9, neginf=-1e9)

        ds = pos5[..., 2]
        ds = _nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0)
        scale = F.softplus(self.dist_scale)
        a = a - scale * ds

        a = a.masked_fill(v_point < 0.5, -1e9)
        w = _safe_softmax(a, dim=-1).unsqueeze(-1)
        w = self.attn_drop(w)
        w = w / (w.sum(dim=-2, keepdim=True) + 1e-6)
        w = _nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        z = (w * p).sum(dim=3)
        z = self.norm(z)
        z = _nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z


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
        e = _nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

        a = self.score(e).squeeze(-1)
        a = _nan_to_num(a, nan=-1e9, posinf=1e9, neginf=-1e9)

        w = _safe_softmax(a, dim=1).unsqueeze(-1)
        c = (w * e).sum(dim=1)
        z = self.proj(c)
        z = self.norm(z)
        z = _nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z


class TriDAMFreq(nn.Module):
    def __init__(self, d_model: int, gate_hidden: int = None, branch_drop_p: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        h = int(gate_hidden) if gate_hidden is not None else max(32, self.d_model // 4)

        self.gate = nn.Sequential(
            nn.Linear(4, h),
            nn.ReLU(),
            nn.Linear(h, 3),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.logit_temp = nn.Parameter(torch.tensor(0.5413248546))
        self.branch_drop_p = float(branch_drop_p)

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

        z_obs = _nan_to_num(z_obs, nan=0.0, posinf=0.0, neginf=0.0)
        z_bg  = _nan_to_num(z_bg,  nan=0.0, posinf=0.0, neginf=0.0)
        z_fc  = _nan_to_num(z_fc,  nan=0.0, posinf=0.0, neginf=0.0)

        o = z_obs.permute(0, 2, 1, 3)
        b = z_bg.permute(0, 2, 1, 3)
        f = z_fc.permute(0, 2, 1, 3)

        o_f = _nan_to_num_complex(torch.fft.rfft(o, dim=2))
        b_f = _nan_to_num_complex(torch.fft.rfft(b, dim=2))
        f_f = _nan_to_num_complex(torch.fft.rfft(f, dim=2))

        mag_o = torch.log1p(o_f.abs().mean(dim=-1))
        mag_b = torch.log1p(b_f.abs().mean(dim=-1))
        mag_f = torch.log1p(f_f.abs().mean(dim=-1))
        mag_o = _nan_to_num(mag_o, nan=0.0, posinf=20.0, neginf=0.0).clamp(0.0, 20.0)
        mag_b = _nan_to_num(mag_b, nan=0.0, posinf=20.0, neginf=0.0).clamp(0.0, 20.0)
        mag_f = _nan_to_num(mag_f, nan=0.0, posinf=20.0, neginf=0.0).clamp(0.0, 20.0)
        mag_m = (mag_o + mag_b + mag_f) / 3.0

        gate_in = torch.stack([mag_o, mag_b, mag_f, mag_m], dim=-1)
        gate_in = _nan_to_num(gate_in, nan=0.0, posinf=0.0, neginf=0.0)

        logits = self.gate(gate_in)
        logits = _nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        temp = F.softplus(self.logit_temp) + 1e-6
        w = _safe_softmax(logits / temp, dim=-1)

        if self.training and self.branch_drop_p > 0.0:
            p = self.branch_drop_p
            trigger = (torch.rand((B, N), device=z_obs.device) < p)
            if trigger.any():
                which = torch.randint(low=0, high=3, size=(B, N), device=z_obs.device)
                mask = torch.ones((B, N, 1, 3), device=z_obs.device, dtype=w.dtype)
                mask.scatter_(-1, which.view(B, N, 1, 1), 0.0)
                mask = torch.where(trigger.view(B, N, 1, 1), mask, torch.ones_like(mask))
                w = w * mask
                w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

        w0 = w[..., 0:1]
        w1 = w[..., 1:2]
        w2 = w[..., 2:3]

        sel_f = w0 * o_f + w1 * b_f + w2 * f_f
        mean_f = (o_f + b_f + f_f) / 3.0

        a = torch.sigmoid(self.alpha)
        y_f = (1.0 - a) * mean_f + a * sel_f
        y_f = _nan_to_num_complex(y_f)

        y = torch.fft.irfft(y_f, n=L, dim=2)
        y = _nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        z = y.permute(0, 2, 1, 3)

        z2 = z.permute(0, 2, 3, 1).reshape(B * N, D, L)
        z2 = self.smooth(z2).reshape(B, N, D, L).permute(0, 3, 1, 2)

        out = self.norm(z + z2)
        out = _nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class FNPFusion(nn.Module):
    """
    Ablation-B: remove the whole vector branch (obs/bg/fc vec encoders + tri_vec + FiLM)
    Keep input shapes unchanged, but ignore vector features.
    """
    def __init__(
        self,
        d_model: int = 128,
        modes: int = 8,
        K_bg: int = 16,
        bg_use_setconv: bool = True,
        fc_use_setconv: bool = True,
        fc_emb_scalar: int = 32,
        obs_setconv_topk: int = None,
        obs_setconv_drop: float = 0.0,
        nfl_spectral_drop: float = 0.0,
        grid_point_drop: float = 0.0,
        grid_attn_drop: float = 0.0,
        tri_branch_drop_p: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(K_bg)
        self.fc_emb_scalar = int(fc_emb_scalar)

        # scalar: [P3, dP3, W3]
        self.obs_scalar_enc = FuncRepVFR(
            in_dim=7, d_model=self.d_model, modes=modes,
            setconv_topk=obs_setconv_topk, setconv_drop=obs_setconv_drop,
            spectral_drop=nfl_spectral_drop
        )

        # analysis scalar (speed only)
        self.bg_scalar_enc = GridEncoder(
            d_model=self.d_model,
            K=self.K,
            feat_per_point=2,
            use_setconv=bool(bg_use_setconv),
            point_drop=grid_point_drop,
            attn_drop=grid_attn_drop,
        )

        # forecast scalar (speed only)
        self.fc_scalar_enc = GridEncoder(
            d_model=self.fc_emb_scalar,
            K=self.K,
            feat_per_point=2,
            use_setconv=bool(fc_use_setconv),
            point_drop=grid_point_drop,
            attn_drop=grid_attn_drop,
        )
        self.fc_scalar_pool = HorizonPool(e_dim=self.fc_emb_scalar, d_model=self.d_model)

        self.tri_scalar = TriDAMFreq(self.d_model, branch_drop_p=tri_branch_drop_p)
        self.out_norm = nn.LayerNorm(self.d_model)

        self._warned_nonfinite = False

    def forward(self, x_obs, x_an, coords, pos, fc0, x_an_valid=None, fc0v=None):
        if (not self._warned_nonfinite):
            for name, t in [("x_obs", x_obs), ("x_an", x_an), ("coords", coords), ("pos", pos), ("fc0", fc0)]:
                if isinstance(t, torch.Tensor) and (not torch.isfinite(t).all()):
                    print(f"[FNPFusion] WARNING: non-finite detected in {name}. nan_to_num safeguards are applied.")
                    self._warned_nonfinite = True
                    break

        x_obs = _nan_to_num(x_obs, nan=0.0, posinf=0.0, neginf=0.0)
        x_an  = _nan_to_num(x_an,  nan=0.0, posinf=0.0, neginf=0.0)
        fc0   = _nan_to_num(fc0,   nan=0.0, posinf=0.0, neginf=0.0)

        if x_an_valid is not None:
            x_an_valid = _nan_to_num(x_an_valid, nan=0.0, posinf=0.0, neginf=0.0)
        if fc0v is not None:
            fc0v = _nan_to_num(fc0v, nan=0.0, posinf=0.0, neginf=0.0)

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

        # ===== scalar branch only =====
        x_obs_scalar = x_obs[..., 0:7]  # [P3, dP3, W3]

        an_speed = dyn_an[..., 0:2]
        an_speed_v = dyn_an_v[..., 0:2] if dyn_an_v is not None else None

        fc0_speed = fc0[..., 0:2]
        fc0_speed_v = fc0v[..., 0:2] if fc0v is not None else None

        z_obs_s = self.obs_scalar_enc(x_obs_scalar, coords)           # (B,L,N,D)
        z_bg_s = self.bg_scalar_enc(an_speed, pos, valid=an_speed_v)  # (B,L,N,D)

        e_fc_s = self.fc_scalar_enc(fc0_speed, pos, valid=fc0_speed_v)  # (B,H,N,Es)
        z_fc_s0 = self.fc_scalar_pool(e_fc_s)                            # (B,N,D)
        z_fc_s = z_fc_s0.unsqueeze(1).expand(B, L, N, self.d_model)       # (B,L,N,D)

        z = self.tri_scalar(z_obs=z_obs_s, z_bg=z_bg_s, z_fc=z_fc_s)
        z = self.out_norm(z)
        z = _nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # IMPORTANT: e_fc is scalar-only now: (B,H,N,fc_emb_scalar)
        return z, e_fc_s
