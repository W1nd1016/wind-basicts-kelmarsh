#models/fnp_fusion_OnlyAnalysis_noRot_paper1_plus_v2_wake_downwind.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用你论文1版不旋转 ablation 文件里已有模块与工具（不动）
from models.fnp_fusion_OnlyAnalysis import (
    _nan_to_num,
    _nan_to_num_complex,
    _safe_softmax,
    FuncRepVFR,
    GridEncoder,
    HorizonPool,
    TriDAMFreq,
    VecFiLM,
    _wind_uv_from_speed_dir_sincos,
)

# -------------------------
# angle helpers (v2-4 optional calibration)
# -------------------------
def _apply_delta_to_sincos(sin_a: torch.Tensor, cos_a: torch.Tensor, delta_rad: torch.Tensor):
    """
    给 angle 加一个小偏置 delta（弧度），但输入是 sin/cos。
    sin(a+δ)=sin a cosδ + cos a sinδ
    cos(a+δ)=cos a cosδ - sin a sinδ
    """
    cd = torch.cos(delta_rad)
    sd = torch.sin(delta_rad)
    sin2 = sin_a * cd + cos_a * sd
    cos2 = cos_a * cd - sin_a * sd
    return sin2, cos2


def _unit_uv(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-6):
    n = torch.sqrt(u * u + v * v + eps)
    return u / n, v / n


def _cos_sin_between_uv(ax, ay, bx, by, eps: float = 1e-6):
    """
    给两个2D向量 a,b，输出：
      cos = dot(a,b)/(|a||b|)
      sin = cross(a,b)/(|a||b|) 其中 cross = ax*by - ay*bx = sin(theta_b - theta_a)
    """
    na = torch.sqrt(ax * ax + ay * ay + eps)
    nb = torch.sqrt(bx * bx + by * by + eps)
    axu, ayu = ax / na, ay / na
    bxu, byu = bx / nb, by / nb
    cos = axu * bxu + ayu * byu
    sin = axu * byu - ayu * bxu
    return cos, sin


# -------------------------
# (A) wake-directed cross-turbine mixing
# -------------------------
class WakeDirectionalMix(nn.Module):
    """
    用场站xy + flow_hat(t)构造上游->下游权重，对 z 做跨风机 mixing（残差+LN）。
    输出 wake_feat 供 ctx 使用。
    """
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
        """
        z: (B,L,N,D)
        xy: (N,2)
        flow_hat: (B,L,2) unit flow direction
        """
        B, L, N, D = z.shape
        device = z.device
        z = _nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        xy = xy.to(device).float()
        delta = xy.view(N, 1, 2) - xy.view(1, N, 2)  # (N,N,2) = i - j
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

        # upstream factor: along>0 表示 i 在 j 的下游（j -> i 有影响）
        upstream = torch.sigmoid((sgn * along) / tau)

        a_pos = F.relu(along)
        w = upstream * torch.exp(-(a_pos / sig_a) ** 2) * torch.exp(-(cross / sig_c) ** 2)

        eye = torch.eye(N, device=device).view(1, 1, N, N)
        w = w * (1.0 - eye)

        wake_strength = w.sum(dim=-1)  # (B,L,N)
        mean_along = (w * a_pos).sum(dim=-1) / (wake_strength + 1e-6)

        W = w / (w.sum(dim=-1, keepdim=True) + 1e-6)  # (B,L,N,N)

        msg = torch.einsum("blij,bljd->blid", W, z)
        msg = self.drop(msg)

        g = torch.sigmoid(self.mix_gate)
        out = self.norm(z + g * msg)

        wake_feat = torch.stack([wake_strength, mean_along], dim=-1)  # (B,L,N,2)
        wake_feat = _nan_to_num(wake_feat, nan=0.0, posinf=0.0, neginf=0.0)
        return out, wake_feat


# -------------------------
# (B) downwind grid penalty: modify pos[...,2]
# -------------------------
class DownwindPosPenalty(nn.Module):
    """
    dist' = dist + lam * relu(sign * (dx*u + dy*v))
    """
    def __init__(self):
        super().__init__()
        self.log_lam = nn.Parameter(torch.tensor(-1.0))    # softplus ~ 0.3
        self.dir_sign = nn.Parameter(torch.tensor(1.5))    # tanh ~ 0.9

    def apply(self, pos5: torch.Tensor, dir_hat: torch.Tensor):
        """
        pos5: (B,T,N,K,3)
        dir_hat: (B,T,N,2) unit direction
        """
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


# -------------------------
# TriDAMFreq with ctx_obg + ctx_bgfc injected into gate input (paper1 scalar fusion kept!)
# -------------------------
class TriDAMFreqWith2Ctx(nn.Module):
    """
    论文1版 TriDAMFreq 的“同结构”扩展：
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
        self.logit_temp = nn.Parameter(torch.tensor(0.5413248546))  # softplus=1
        self.branch_drop_p = float(branch_drop_p)

        self.smooth = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(self.d_model)

    def _ctx_mag(self, ctx: torch.Tensor, L: int):
        """
        ctx: (B,L,N,C) real
        return mag: (B,N,F)
        """
        if ctx is None:
            return None
        ctx = _nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        c = ctx.permute(0, 2, 1, 3)  # (B,N,L,C)
        c_f = _nan_to_num_complex(torch.fft.rfft(c, dim=2))
        mag = torch.log1p(c_f.abs().mean(dim=-1)).clamp(0.0, 20.0)  # (B,N,F)
        return mag

    def forward(self, z_obs: torch.Tensor, z_bg: torch.Tensor, z_fc: torch.Tensor, ctx_obg=None, ctx_bgfc=None):
        if z_obs.shape != z_bg.shape or z_obs.shape != z_fc.shape:
            raise RuntimeError(f"TriDAMFreqWith2Ctx expects same shapes, got obs={tuple(z_obs.shape)} bg={tuple(z_bg.shape)} fc={tuple(z_fc.shape)}")
        B, L, N, D = z_obs.shape
        if D != self.d_model:
            raise RuntimeError(f"d_model mismatch: got D={D}, expected {self.d_model}")

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

        mag_obg = self._ctx_mag(ctx_obg, L)
        mag_bgfc = self._ctx_mag(ctx_bgfc, L)
        if mag_obg is None:
            mag_obg = torch.zeros_like(mag_o)
        if mag_bgfc is None:
            mag_bgfc = torch.zeros_like(mag_o)

        gate_in = torch.stack([mag_o, mag_b, mag_f, mag_obg, mag_bgfc, mag_m], dim=-1)  # (B,N,F,6)
        gate_in = _nan_to_num(gate_in, nan=0.0, posinf=0.0, neginf=0.0)

        logits = self.gate(gate_in)  # (B,N,F,3)
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

        sel_f = w[..., 0:1] * o_f + w[..., 1:2] * b_f + w[..., 2:3] * f_f
        mean_f = (o_f + b_f + f_f) / 3.0

        a = torch.sigmoid(self.alpha)
        y_f = (1.0 - a) * mean_f + a * sel_f
        y_f = _nan_to_num_complex(y_f)

        y = torch.fft.irfft(y_f, n=L, dim=2)
        y = _nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        z = y.permute(0, 2, 1, 3)  # (B,L,N,D)

        z2 = z.permute(0, 2, 3, 1).reshape(B * N, D, L)
        z2 = self.smooth(z2).reshape(B, N, D, L).permute(0, 3, 1, 2)
        z2 = _nan_to_num(z2, nan=0.0, posinf=0.0, neginf=0.0)

        out = self.norm(z + z2)
        return _nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------------
# Main: paper1 no-rotation + v2(4pts) + wake + downwind penalty
# -------------------------
class FNPFusionOnlyAnalysis_NoRot_Paper1PlusV2(nn.Module):
    """
    保持论文1版“不旋转版”的主结构不变：
      scalar: obs/bg/fc -> (TriDAMFreq 变体：加 ctx_obg/ctx_bgfc 到 gate)
      vector: obs/bg/fc -> TriDAMFreq（原版）
      vec_film: 用 vector 分支调制 scalar 分支输出

    v2 四点 + (A)(B)：
      1) obs_vec 输入加 nac_sin/nac_cos + cos_mis/sin_mis
      2) ctx_obg 加 cos_sa/sin_sa + cos_mis/sin_mis（可选再拼 wake）
      3) ctx_bgfc 加 cos_af/sin_af
      4) 可选角度小校准 delta_wdir/delta_nac（默认关闭）

      (A) wake mixing: 对 z_obs_s/z_obs_v 做跨风机方向 mixing + wake_feat 提供给 ctx_obg
      (B) downwind penalty: 修改 pos[...,2] 再喂给 GridEncoder
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
        film_dropout_p: float = 0.05,
        # toggles
        use_wake: bool = True,
        use_downwind_penalty: bool = True,
        use_angle_calib: bool = False,
        max_calib_deg: float = 10.0,
        scalar_branch_drop_p: float = 0.0,
        # ablation knobs (keep same style)
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

        self.use_wake = bool(use_wake)
        self.use_downwind_penalty = bool(use_downwind_penalty)
        self.use_angle_calib = bool(use_angle_calib)
        self.max_calib_rad = float(max_calib_deg) * 3.141592653589793 / 180.0

        # --- scalar encoders ---
        self.obs_scalar_enc = FuncRepVFR(
            in_dim=3, d_model=self.d_model, modes=modes,
            setconv_topk=obs_setconv_topk, setconv_drop=obs_setconv_drop,
            spectral_drop=nfl_spectral_drop
        )
        self.bg_scalar_enc = GridEncoder(
            d_model=self.d_model, K=self.K, feat_per_point=1,
            use_setconv=bool(bg_use_setconv),
            point_drop=grid_point_drop, attn_drop=grid_attn_drop,
        )
        self.fc_scalar_enc = GridEncoder(
            d_model=int(fc_emb_scalar), K=self.K, feat_per_point=1,
            use_setconv=bool(fc_use_setconv),
            point_drop=grid_point_drop, attn_drop=grid_attn_drop,
        )
        self.fc_scalar_pool = HorizonPool(e_dim=int(fc_emb_scalar), d_model=self.d_model)

        # --- vector encoders ---
        # obs_vec: [u,v,nac_sin,nac_cos,cos_mis,sin_mis] -> 6 dims
        self.obs_vec_enc = FuncRepVFR(
            in_dim=6, d_model=self.d_model, modes=modes,
            setconv_topk=obs_setconv_topk, setconv_drop=obs_setconv_drop,
            spectral_drop=nfl_spectral_drop
        )
        self.bg_vec_enc = GridEncoder(
            d_model=self.d_model, K=self.K, feat_per_point=2,
            use_setconv=bool(bg_use_setconv),
            point_drop=grid_point_drop, attn_drop=grid_attn_drop,
        )
        self.fc_vec_enc = GridEncoder(
            d_model=int(fc_emb_vec), K=self.K, feat_per_point=2,
            use_setconv=bool(fc_use_setconv),
            point_drop=grid_point_drop, attn_drop=grid_attn_drop,
        )
        self.fc_vec_pool = HorizonPool(e_dim=int(fc_emb_vec), d_model=self.d_model)

        # --- scalar fusion: paper1 tri fusion but with ctx injected into gate ---
        self.tri_scalar = TriDAMFreqWith2Ctx(self.d_model, branch_drop_p=float(scalar_branch_drop_p))

        # --- vector fusion: keep paper1 TriDAMFreq unchanged ---
        self.tri_vec = TriDAMFreq(self.d_model, branch_drop_p=float(tri_branch_drop_p))

        # --- vec_film keep unchanged ---
        self.vec_film = VecFiLM(self.d_model, dropout_p=film_dropout_p)
        self.film_scale = nn.Parameter(torch.tensor(-2.0))
        self.out_norm = nn.LayerNorm(self.d_model)

        # --- (A)(B) modules ---
        self.wake_mix = WakeDirectionalMix(self.d_model, dropout_p=0.0)
        self.pos_pen = DownwindPosPenalty()

        # --- v2-4 optional calibration (default off) ---
        self.delta_wdir = nn.Parameter(torch.tensor(0.0))
        self.delta_nac = nn.Parameter(torch.tensor(0.0))

    def _broadcast_pos(self, pos: torch.Tensor, B: int, T: int, N: int, K: int, device):
        if pos.ndim == 3:
            pos5 = pos.unsqueeze(0).unsqueeze(0)  # (1,1,N,K,3)
        elif pos.ndim == 4:
            pos5 = pos.unsqueeze(1)               # (B,1,N,K,3)
        elif pos.ndim == 5:
            pos5 = pos
        else:
            raise RuntimeError(f"pos must be (N,K,3)/(B,N,K,3)/(B,T,N,K,3), got {tuple(pos.shape)}")

        pos5 = pos5.to(device)
        if pos5.size(0) == 1 and B > 1:
            pos5 = pos5.expand(B, -1, -1, -1, -1)
        if pos5.size(1) == 1 and T > 1:
            pos5 = pos5.expand(-1, T, -1, -1, -1)
        if pos5.size(2) != N or pos5.size(3) != K:
            raise RuntimeError(f"pos broadcast mismatch: got {tuple(pos5.shape)} expected (B,T,N,K,3)=({B},{T},{N},{K},3)")
        return pos5

    def forward(self, x_obs, x_an, coords, pos, fc0, x_an_valid=None, fc0v=None):
        x_obs = _nan_to_num(x_obs, nan=0.0, posinf=0.0, neginf=0.0)
        x_an  = _nan_to_num(x_an,  nan=0.0, posinf=0.0, neginf=0.0)
        fc0   = _nan_to_num(fc0,   nan=0.0, posinf=0.0, neginf=0.0)
        if x_an_valid is not None:
            x_an_valid = _nan_to_num(x_an_valid, nan=0.0, posinf=0.0, neginf=0.0)
        if fc0v is not None:
            fc0v = _nan_to_num(fc0v, nan=0.0, posinf=0.0, neginf=0.0)

        if x_obs.ndim != 4 or x_obs.shape[-1] != 7:
            raise RuntimeError(f"x_obs must be (B,L,N,7), got {tuple(x_obs.shape)}")
        if x_an.ndim != 4:
            raise RuntimeError(f"x_an must be (B,L,N,K*4), got {tuple(x_an.shape)}")
        if fc0.ndim != 5 or fc0.shape[-1] != 4:
            raise RuntimeError(f"fc0 must be (B,H,N,K,4), got {tuple(fc0.shape)}")

        B, L, N, _ = x_obs.shape
        _, H, N2, K, _ = fc0.shape
        if N2 != N or K != self.K:
            raise RuntimeError(f"fc0 shape mismatch: expected N={N},K={self.K}, got {tuple(fc0.shape)}")

        dyn_an = x_an.view(B, L, N, self.K, 4)
        dyn_an_v = x_an_valid.view(B, L, N, self.K, 4) if x_an_valid is not None else None

        # -------- coords -> xy for wake (use first time slice) --------
        coords = coords.to(x_obs.device)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != L * N:
            raise RuntimeError(f"coords must be (L*N,3) to match FuncRepVFR, got {tuple(coords.shape)} but L*N={L*N}")
        xy = coords.view(L, N, 3)[0, :, 1:3].detach()  # (N,2)

        # =========================
        # 0) SCADA wind dir & nac dir (with optional small calibration)
        # =========================
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

        # misalignment Δψ = wind_dir - nac_dir
        cos_mis = dir_cos * nac_cos + dir_sin * nac_sin
        sin_mis = dir_sin * nac_cos - dir_cos * nac_sin
        cos_mis = _nan_to_num(cos_mis, nan=0.0, posinf=0.0, neginf=0.0)
        sin_mis = _nan_to_num(sin_mis, nan=0.0, posinf=0.0, neginf=0.0)

        # SCADA uv in earth frame (no rotation!)
        u_sc, v_sc = _wind_uv_from_speed_dir_sincos(W3, dir_sin, dir_cos)

        # flow_hat for wake: farm-mean unit direction (B,L,2)
        u_hat, v_hat = _unit_uv(u_sc, v_sc)
        u_mean = u_hat.mean(dim=2)
        v_mean = v_hat.mean(dim=2)
        u_mean, v_mean = _unit_uv(u_mean, v_mean)
        flow_hat = torch.stack([u_mean, v_mean], dim=-1)

        # =========================
        # 1) encode SCADA scalar / vector
        # =========================
        x_obs_scalar = x_obs[..., 0:3]  # [P3,dP3,W3]
        z_obs_s = self.obs_scalar_enc(x_obs_scalar, coords)  # (B,L,N,D)

        # obs_vec: [u,v,nac_sin,nac_cos,cos_mis,sin_mis]
        x_obs_vec = torch.stack([u_sc, v_sc, nac_sin,nac_cos, cos_mis, sin_mis], dim=-1)
        z_obs_v = self.obs_vec_enc(x_obs_vec, coords)  # (B,L,N,D)

        # (A) wake mixing
        wake_feat = None
        if self.use_wake:
            z_obs_s, wake_feat = self.wake_mix(z_obs_s, xy=xy, flow_hat=flow_hat)
            z_obs_v, _ = self.wake_mix(z_obs_v, xy=xy, flow_hat=flow_hat)

        # =========================
        # 2) encode CERRA analysis scalar/vector (with optional downwind penalty)
        # =========================
        an_speed = dyn_an[..., 0:1]  # (B,L,N,K,1)
        an_speed_v = dyn_an_v[..., 0:1] if dyn_an_v is not None else None

        an_vec = dyn_an[..., 2:4]    # (B,L,N,K,2) [u,v]
        an_uv_v = dyn_an_v[..., 2:4] if dyn_an_v is not None else None

        pos_an = self._broadcast_pos(pos, B=B, T=L, N=N, K=self.K, device=x_obs.device)

        # 用 analysis 的 mean uv 做下风惩罚方向
        if dyn_an_v is not None:
            vpt = (an_uv_v.mean(dim=-1) > 0.5).float()  # (B,L,N,K)
            u_anm = (an_vec[..., 0] * vpt).sum(dim=3) / (vpt.sum(dim=3) + 1e-6)
            v_anm = (an_vec[..., 1] * vpt).sum(dim=3) / (vpt.sum(dim=3) + 1e-6)
        else:
            u_anm = an_vec[..., 0].mean(dim=3)
            v_anm = an_vec[..., 1].mean(dim=3)
        u_anm, v_anm = _unit_uv(u_anm, v_anm)
        an_dir = torch.stack([u_anm, v_anm], dim=-1)  # (B,L,N,2)

        if self.use_downwind_penalty:
            pos_an = self.pos_pen.apply(pos_an, an_dir)

        z_bg_s = self.bg_scalar_enc(an_speed, pos_an, valid=an_speed_v)  # (B,L,N,D)
        z_bg_v = self.bg_vec_enc(an_vec,   pos_an, valid=an_uv_v)        # (B,L,N,D)

        # =========================
        # 3) encode CERRA forecast scalar/vector (with optional downwind penalty)
        # =========================
        fc_speed = fc0[..., 0:1]     # (B,H,N,K,1)
        fc_speed_v = fc0v[..., 0:1] if fc0v is not None else None

        fc_vec = fc0[..., 2:4]       # (B,H,N,K,2)
        fc_uv_v = fc0v[..., 2:4] if fc0v is not None else None

        pos_fc = self._broadcast_pos(pos, B=B, T=H, N=N, K=self.K, device=x_obs.device)

        # forecast direction per lead (mean over K) for downwind penalty
        if fc_uv_v is not None:
            vpt = (fc_uv_v.mean(dim=-1) > 0.5).float()  # (B,H,N,K)
            u_fcm = (fc_vec[..., 0] * vpt).sum(dim=3) / (vpt.sum(dim=3) + 1e-6)
            v_fcm = (fc_vec[..., 1] * vpt).sum(dim=3) / (vpt.sum(dim=3) + 1e-6)
        else:
            u_fcm = fc_vec[..., 0].mean(dim=3)
            v_fcm = fc_vec[..., 1].mean(dim=3)
        u_fcm, v_fcm = _unit_uv(u_fcm, v_fcm)
        fc_dir = torch.stack([u_fcm, v_fcm], dim=-1)  # (B,H,N,2)

        if self.use_downwind_penalty:
            pos_fc = self.pos_pen.apply(pos_fc, fc_dir)

        e_fc_s = self.fc_scalar_enc(fc_speed, pos_fc, valid=fc_speed_v)  # (B,H,N,Es)
        z_fc_s0 = self.fc_scalar_pool(e_fc_s)                            # (B,N,D)
        z_fc_s = z_fc_s0.unsqueeze(1).expand(B, L, N, self.d_model)       # (B,L,N,D)

        e_fc_v = self.fc_vec_enc(fc_vec, pos_fc, valid=fc_uv_v)          # (B,H,N,Ev)
        z_fc_v0 = self.fc_vec_pool(e_fc_v)                               # (B,N,D)
        z_fc_v = z_fc_v0.unsqueeze(1).expand(B, L, N, self.d_model)       # (B,L,N,D)

        # =========================
        # 4) v2-2: ctx_obg (SCADA ↔ analysis) alignment + misalignment (+ optional wake)
        # =========================
        # cos_sa/sin_sa: scada uv vs analysis mean uv
        cos_sa, sin_sa = _cos_sin_between_uv(u_sc, v_sc, u_anm, v_anm)

        # ctx_obg channels: [cos_sa, sin_sa, cos_mis, sin_mis] (+ wake_strength, mean_along)
        ctx_obg = torch.stack([cos_sa, sin_sa, cos_mis, sin_mis], dim=-1)  # (B,L,N,4)
        if self.use_wake and wake_feat is not None:
            ctx_obg = torch.cat([ctx_obg, wake_feat], dim=-1)  # (B,L,N,6)

        # =========================
        # 5) v2-3: ctx_bgfc (analysis(t0) ↔ forecast mean) alignment
        # =========================
        # analysis(t0) mean dir: take last history step
        u_an_t0 = u_anm[:, -1]  # (B,N)
        v_an_t0 = v_anm[:, -1]  # (B,N)

        # forecast mean dir: mean over H leads
        u_fc_mean = u_fcm.mean(dim=1)  # (B,N)
        v_fc_mean = v_fcm.mean(dim=1)  # (B,N)

        cos_af, sin_af = _cos_sin_between_uv(u_an_t0, v_an_t0, u_fc_mean, v_fc_mean)  # (B,N)
        # expand to (B,L,N,2)
        cos_af = cos_af.unsqueeze(1).expand(B, L, N)
        sin_af = sin_af.unsqueeze(1).expand(B, L, N)
        ctx_bgfc = torch.stack([cos_af, sin_af], dim=-1)  # (B,L,N,2)

        # =========================
        # 6) scalar fusion (paper1 tri fusion kept, but gate gets ctx magnitudes)
        # =========================
        z_scalar = self.tri_scalar(z_obs=z_obs_s, z_bg=z_bg_s, z_fc=z_fc_s, ctx_obg=ctx_obg, ctx_bgfc=ctx_bgfc)

        # =========================
        # 7) vector fusion (paper1 keep) + vec_film modulate scalar (paper1 keep)
        # =========================
        z_vec = self.tri_vec(z_obs=z_obs_v, z_bg=z_bg_v, z_fc=z_fc_v)  # (B,L,N,D)

        gamma, beta = self.vec_film(z_vec)
        s = torch.sigmoid(self.film_scale)  
        z = (1.0 + s * gamma) * z_scalar + (s * beta)
        z = self.out_norm(z)
        z = _nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # decoder exog unchanged
        e_fc = torch.cat([e_fc_s, e_fc_v], dim=-1)  # (B,H,N,Es+Ev)
        return z, e_fc
