import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fnp_fusion_OnlyAnalysis import (
    _nan_to_num,
    _nan_to_num_complex,
    _safe_softmax,
    FuncRepVFR,
    GridEncoder,
    HorizonPool,
    TriDAMFreq,
    VecFiLM,  # kept imported but not used (ok)
    _wind_uv_from_speed_dir_sincos,
)

def _apply_delta_to_sincos(sin_a: torch.Tensor, cos_a: torch.Tensor, delta_rad: torch.Tensor):
    cd = torch.cos(delta_rad)
    sd = torch.sin(delta_rad)
    sin2 = sin_a * cd + cos_a * sd
    cos2 = cos_a * cd - sin_a * sd
    return sin2, cos2

def _unit_uv(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-6):
    n = torch.sqrt(u * u + v * v + eps)
    return u / n, v / n

def _cos_sin_between_uv(ax, ay, bx, by, eps: float = 1e-6):
    na = torch.sqrt(ax * ax + ay * ay + eps)
    nb = torch.sqrt(bx * bx + by * by + eps)
    axu, ayu = ax / na, ay / na
    bxu, byu = bx / nb, by / nb
    cos = axu * bxu + ayu * byu
    sin = axu * byu - ayu * bxu
    return cos, sin


class WakeDirectionalMix(nn.Module):
    def __init__(self, d_model: int, dropout_p: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.log_sig_along = nn.Parameter(torch.tensor(0.0))
        self.log_sig_cross = nn.Parameter(torch.tensor(0.0))
        self.log_tau = nn.Parameter(torch.tensor(-0.7))
        self.dir_sign = nn.Parameter(torch.tensor(1.5))
        self.mix_gate = nn.Parameter(torch.tensor(-2.0))
        self.drop = nn.Dropout(p=float(dropout_p))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, z: torch.Tensor, xy: torch.Tensor, flow_hat: torch.Tensor):
        B, L, N, D = z.shape
        device = z.device
        z = _nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        xy = xy.to(device).float()
        delta = xy.view(N, 1, 2) - xy.view(1, N, 2)
        dx = delta[..., 0]
        dy = delta[..., 1]

        fu = flow_hat[..., 0].view(B, L, 1, 1)
        fv = flow_hat[..., 1].view(B, L, 1, 1)

        along = dx.view(1, 1, N, N) * fu + dy.view(1, 1, N, N) * fv
        cross = torch.abs(dx.view(1, 1, N, N) * fv - dy.view(1, 1, N, N) * fu)

        sig_a = torch.exp(self.log_sig_along) + 1e-6
        sig_c = torch.exp(self.log_sig_cross) + 1e-6
        tau = torch.exp(self.log_tau) + 1e-6
        sgn = torch.tanh(self.dir_sign)

        upstream = torch.sigmoid((sgn * along) / tau)
        a_pos = F.relu(along)

        w = upstream * torch.exp(-(a_pos / sig_a) ** 2) * torch.exp(-(cross / sig_c) ** 2)
        eye = torch.eye(N, device=device).view(1, 1, N, N)
        w = w * (1.0 - eye)

        wake_strength = w.sum(dim=-1)
        mean_along = (w * a_pos).sum(dim=-1) / (wake_strength + 1e-6)

        W = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        msg = torch.einsum("blij,bljd->blid", W, z)
        msg = self.drop(msg)

        g = torch.sigmoid(self.mix_gate)
        out = self.norm(z + g * msg)

        wake_feat = torch.stack([wake_strength, mean_along], dim=-1)
        wake_feat = _nan_to_num(wake_feat, nan=0.0, posinf=0.0, neginf=0.0)
        return out, wake_feat


class DownwindPosPenalty(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_lam = nn.Parameter(torch.tensor(-1.0))
        self.dir_sign = nn.Parameter(torch.tensor(1.5))

    def apply(self, pos5: torch.Tensor, dir_hat: torch.Tensor):
        pos5 = _nan_to_num(pos5, nan=0.0, posinf=0.0, neginf=0.0)
        dir_hat = _nan_to_num(dir_hat, nan=0.0, posinf=0.0, neginf=0.0)

        dx = pos5[..., 0]
        dy = pos5[..., 1]
        ds = pos5[..., 2]

        du = dir_hat[..., 0].unsqueeze(-1)
        dv = dir_hat[..., 1].unsqueeze(-1)
        dot = dx * du + dy * dv

        lam = F.softplus(self.log_lam)
        sgn = torch.tanh(self.dir_sign)
        pen = lam * F.relu(sgn * dot)

        out = pos5.clone()
        out[..., 2] = ds + pen
        return out


class TriDAMFreqWith2Ctx(nn.Module):
    """
    scalar tri-fusion with ctx_obg + ctx_bgfc injected into gate.
    gate_in: [mag_obs, mag_bg, mag_fc, mag_ctx_obg, mag_ctx_bgfc, mag_mean]
    """
    def __init__(self, d_model: int, gate_hidden: int = None, branch_drop_p: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        h = int(gate_hidden) if gate_hidden is not None else max(32, self.d_model // 4)

        self.gate = nn.Sequential(
            nn.Linear(6, h),
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

    def _ctx_mag(self, ctx: torch.Tensor, L: int):
        if ctx is None:
            return None
        ctx = _nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        c = ctx.permute(0, 2, 1, 3)  # (B,N,L,C)
        c_f = _nan_to_num_complex(torch.fft.rfft(c, dim=2))
        mag = torch.log1p(c_f.abs().mean(dim=-1)).clamp(0.0, 20.0)
        return mag

    def forward(self, z_obs: torch.Tensor, z_bg: torch.Tensor, z_fc: torch.Tensor, ctx_obg=None, ctx_bgfc=None):
        if z_obs.shape != z_bg.shape or z_obs.shape != z_fc.shape:
            raise RuntimeError("TriDAMFreqWith2Ctx expects same shapes for three branches.")
        B, L, N, D = z_obs.shape
        if D != self.d_model:
            raise RuntimeError("d_model mismatch in TriDAMFreqWith2Ctx.")

        z_obs = _nan_to_num(z_obs, nan=0.0, posinf=0.0, neginf=0.0)
        z_bg  = _nan_to_num(z_bg,  nan=0.0, posinf=0.0, neginf=0.0)
        z_fc  = _nan_to_num(z_fc,  nan=0.0, posinf=0.0, neginf=0.0)

        o = z_obs.permute(0, 2, 1, 3)  # (B,N,L,D)
        b = z_bg.permute(0, 2, 1, 3)
        f = z_fc.permute(0, 2, 1, 3)

        o_f = _nan_to_num_complex(torch.fft.rfft(o, dim=2))
        b_f = _nan_to_num_complex(torch.fft.rfft(b, dim=2))
        f_f = _nan_to_num_complex(torch.fft.rfft(f, dim=2))

        mag_o = torch.log1p(o_f.abs().mean(dim=-1)).clamp(0.0, 20.0)
        mag_b = torch.log1p(b_f.abs().mean(dim=-1)).clamp(0.0, 20.0)
        mag_f = torch.log1p(f_f.abs().mean(dim=-1)).clamp(0.0, 20.0)
        mag_m = (mag_o + mag_b + mag_f) / 3.0

        mag_obg = self._ctx_mag(ctx_obg, L) or torch.zeros_like(mag_o)
        mag_bgfc = self._ctx_mag(ctx_bgfc, L) or torch.zeros_like(mag_o)

        gate_in = torch.stack([mag_o, mag_b, mag_f, mag_obg, mag_bgfc, mag_m], dim=-1)  # (B,N,F,6)

        logits = self.gate(gate_in)
        logits = _nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        temp = F.softplus(self.logit_temp) + 1e-6
        w = _safe_softmax(logits / temp, dim=-1)  # (B,N,F,3)

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

        w0, w1, w2 = w[..., 0:1], w[..., 1:2], w[..., 2:3]
        sel_f = w0 * o_f + w1 * b_f + w2 * f_f
        mean_f = (o_f + b_f + f_f) / 3.0

        a = torch.sigmoid(self.alpha)
        y_f = (1.0 - a) * mean_f + a * sel_f
        y_f = _nan_to_num_complex(y_f)

        y = torch.fft.irfft(y_f, n=L, dim=2)
        y = _nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        z = y.permute(0, 2, 1, 3)  # (B,L,N,D)
        z2 = z.permute(0, 2, 3, 1).reshape(B * N, D, L)
        z2 = self.smooth(z2).reshape(B, N, D, L).permute(0, 3, 1, 2)

        out = self.norm(z + z2)
        return _nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class FNPFusionOnlyAnalysis_NoRot_Paper1PlusV2_AbMergeVec(nn.Module):
    """
    Ablation: remove (1) vector separate modeling and (2) VecFiLM modulation.
    Merge vector information into scalar branch, keep tri-fusion + contexts + wake + downwind penalty.

    Option A: keep decoder exog_dim=64 by setting forecast embedding E=64.
    """
    def __init__(
        self,
        d_model: int = 128,
        modes: int = 8,
        K_bg: int = 16,
        bg_use_setconv: bool = True,
        fc_use_setconv: bool = True,
        film_dropout_p: float = 0.05,  # unused but kept for signature compatibility
        use_wake: bool = True,
        use_downwind_penalty: bool = True,
        use_angle_calib: bool = False,
        max_calib_deg: float = 10.0,
        scalar_branch_drop_p: float = 0.0,
        obs_setconv_topk: int = None,
        obs_setconv_drop: float = 0.0,
        nfl_spectral_drop: float = 0.0,
        grid_point_drop: float = 0.0,
        grid_attn_drop: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(K_bg)

        self.use_wake = bool(use_wake)
        self.use_downwind_penalty = bool(use_downwind_penalty)
        self.use_angle_calib = bool(use_angle_calib)
        self.max_calib_rad = float(max_calib_deg) * 3.141592653589793 / 180.0

        # ---------- encoders ----------
        # SCADA scalar now merges vector cues: [P3,dP3,W3,u_sc,v_sc,cos_mis,sin_mis] -> 7 dims
        self.obs_scalar_enc = FuncRepVFR(
            in_dim=7,
            d_model=self.d_model,
            modes=modes,
            setconv_topk=obs_setconv_topk,
            setconv_drop=obs_setconv_drop,
            spectral_drop=nfl_spectral_drop,
        )

        # Analysis scalar now includes [speed,u,v] per grid point -> fp=3
        self.bg_scalar_enc = GridEncoder(
            d_model=self.d_model,
            K=self.K,
            feat_per_point=3,
            use_setconv=bool(bg_use_setconv),
            point_drop=grid_point_drop,
            attn_drop=grid_attn_drop,
        )

        # Forecast scalar embedding set to 64 to keep exog_dim=64
        self.fc_emb = 64
        self.fc_scalar_enc = GridEncoder(
            d_model=self.fc_emb,
            K=self.K,
            feat_per_point=3,  # [speed,u,v]
            use_setconv=bool(fc_use_setconv),
            point_drop=grid_point_drop,
            attn_drop=grid_attn_drop,
        )
        self.fc_scalar_pool = HorizonPool(e_dim=self.fc_emb, d_model=self.d_model)

        # tri-fusion (with ctx injection unchanged)
        self.tri_scalar = TriDAMFreqWith2Ctx(self.d_model, branch_drop_p=float(scalar_branch_drop_p))
        self.out_norm = nn.LayerNorm(self.d_model)

        # wake + downwind penalty unchanged
        self.wake_mix = WakeDirectionalMix(self.d_model, dropout_p=0.0)
        self.pos_pen = DownwindPosPenalty()

        # optional calibration params kept
        self.delta_wdir = nn.Parameter(torch.tensor(0.0))
        self.delta_nac = nn.Parameter(torch.tensor(0.0))

    def _broadcast_pos(self, pos: torch.Tensor, B: int, T: int, N: int, K: int, device):
        if pos.ndim == 3:
            pos5 = pos.unsqueeze(0).unsqueeze(0)
        elif pos.ndim == 4:
            pos5 = pos.unsqueeze(1)
        elif pos.ndim == 5:
            pos5 = pos
        else:
            raise RuntimeError(f"pos must be (N,K,3)/(B,N,K,3)/(B,T,N,K,3), got {tuple(pos.shape)}")

        pos5 = pos5.to(device)
        if pos5.size(0) == 1 and B > 1:
            pos5 = pos5.expand(B, -1, -1, -1, -1)
        if pos5.size(1) == 1 and T > 1:
            pos5 = pos5.expand(-1, T, -1, -1, -1)
        return pos5

    def forward(self, x_obs, x_an, coords, pos, fc0, x_an_valid=None, fc0v=None):
        x_obs = _nan_to_num(x_obs, nan=0.0, posinf=0.0, neginf=0.0)
        x_an  = _nan_to_num(x_an,  nan=0.0, posinf=0.0, neginf=0.0)
        fc0   = _nan_to_num(fc0,   nan=0.0, posinf=0.0, neginf=0.0)
        if x_an_valid is not None:
            x_an_valid = _nan_to_num(x_an_valid, nan=0.0, posinf=0.0, neginf=0.0)
        if fc0v is not None:
            fc0v = _nan_to_num(fc0v, nan=0.0, posinf=0.0, neginf=0.0)

        B, L, N, Fobs = x_obs.shape
        _, H, N2, K, fp = fc0.shape
        if N2 != N or K != self.K or fp != 4 or Fobs != 7:
            raise RuntimeError("Input shapes mismatch in AbMergeVec model.")

        # coords -> xy for wake (same assumption as your official code)
        coords = coords.to(x_obs.device)
        xy = coords.view(L, N, 3)[0, :, 1:3].detach()

        # --- SCADA angles and optional calibration ---
        W3      = x_obs[..., 2]
        dir_sin = x_obs[..., 3]
        dir_cos = x_obs[..., 4]
        nac_sin = x_obs[..., 5]
        nac_cos = x_obs[..., 6]

        if self.use_angle_calib:
            dw = self.max_calib_rad * torch.tanh(self.delta_wdir)
            dn = self.max_calib_rad * torch.tanh(self.delta_nac)
            dir_sin, dir_cos = _apply_delta_to_sincos(dir_sin, dir_cos, dw)
            nac_sin, nac_cos = _apply_delta_to_sincos(nac_sin, nac_cos, dn)

        cos_mis = dir_cos * nac_cos + dir_sin * nac_sin
        sin_mis = dir_sin * nac_cos - dir_cos * nac_sin
        cos_mis = _nan_to_num(cos_mis, nan=0.0, posinf=0.0, neginf=0.0)
        sin_mis = _nan_to_num(sin_mis, nan=0.0, posinf=0.0, neginf=0.0)

        # SCADA uv (earth frame)
        u_sc, v_sc = _wind_uv_from_speed_dir_sincos(W3, dir_sin, dir_cos)

        # flow_hat for wake
        u_hat, v_hat = _unit_uv(u_sc, v_sc)
        u_mean = u_hat.mean(dim=2)
        v_mean = v_hat.mean(dim=2)
        u_mean, v_mean = _unit_uv(u_mean, v_mean)
        flow_hat = torch.stack([u_mean, v_mean], dim=-1)

        # --- merged SCADA scalar input ---
        P3 = x_obs[..., 0]
        dP3 = x_obs[..., 1]
        x_obs_scalar = torch.stack([P3, dP3, W3, u_sc, v_sc, nac_cos, nac_sin], dim=-1)  # (B,L,N,7)
        z_obs_s = self.obs_scalar_enc(x_obs_scalar, coords)  # (B,L,N,D)

        # wake mixing on scalar latent only (vector branch removed)
        wake_feat = None
        if self.use_wake:
            z_obs_s, wake_feat = self.wake_mix(z_obs_s, xy=xy, flow_hat=flow_hat)

        # --- analysis: merge speed+uv into scalar grid features ---
        dyn_an = x_an.view(B, L, N, self.K, 4)
        dyn_an_v = x_an_valid.view(B, L, N, self.K, 4) if x_an_valid is not None else None

        an_speed = dyn_an[..., 0:1]  # (B,L,N,K,1)
        an_uv = dyn_an[..., 2:4]     # (B,L,N,K,2)
        an_feat = torch.cat([an_speed, an_uv], dim=-1)  # (B,L,N,K,3)

        # valid mask for [speed,u,v]
        if dyn_an_v is not None:
            vv = torch.cat([dyn_an_v[..., 0:1], dyn_an_v[..., 2:4]], dim=-1)  # (B,L,N,K,3)
        else:
            vv = None

        pos_an = self._broadcast_pos(pos, B=B, T=L, N=N, K=self.K, device=x_obs.device)



        z_bg_s = self.bg_scalar_enc(an_feat, pos_an, valid=vv)  # (B,L,N,D)

        # --- forecast: merge speed+uv into scalar grid features ---
        fc_speed = fc0[..., 0:1]   # (B,H,N,K,1)
        fc_uv = fc0[..., 2:4]      # (B,H,N,K,2)
        fc_feat = torch.cat([fc_speed, fc_uv], dim=-1)  # (B,H,N,K,3)

        if fc0v is not None:
            vv_fc = torch.cat([fc0v[..., 0:1], fc0v[..., 2:4]], dim=-1)  # (B,H,N,K,3)
        else:
            vv_fc = None

        pos_fc = self._broadcast_pos(pos, B=B, T=H, N=N, K=self.K, device=x_obs.device)

        

        # forecast embedding (E=64)
        e_fc_s = self.fc_scalar_enc(fc_feat, pos_fc, valid=vv_fc)  # (B,H,N,64)

        # pooled forecast summary for tri-fusion (to D)
        z_fc_s0 = self.fc_scalar_pool(e_fc_s)  # (B,N,D)
        z_fc_s = z_fc_s0.unsqueeze(1).expand(B, L, N, self.d_model)

        

        # --- scalar tri-fusion (unchanged) ---
        z_scalar = self.tri_scalar(z_obs=z_obs_s, z_bg=z_bg_s, z_fc=z_fc_s)  # (B,L,N,D)

        z = self.out_norm(z_scalar)
        z = _nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # exog for decoder: keep 64 dims
        e_fc = e_fc_s  # (B,H,N,64)
        return z, e_fc
