import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用论文1版模块（不动）
from models.fnp_fusion_OnlyAnalysis import (
    _nan_to_num,
    _nan_to_num_complex,
    _safe_softmax,
    FuncRepVFR,
    GridEncoder,
    HorizonPool,
    VecFiLM,
    _wind_uv_from_speed_dir_sincos,
    _rotate_uv_to_nacelle_frame,
)


def _rotate_sincos_by_delta(sin_t: torch.Tensor, cos_t: torch.Tensor, delta: torch.Tensor):
    """
    rotate angle: theta' = theta + delta
    given (sin,cos) -> (sin',cos')
    """
    # delta is scalar parameter; broadcast ok
    sd = torch.sin(delta)
    cd = torch.cos(delta)
    sin2 = sin_t * cd + cos_t * sd
    cos2 = cos_t * cd - sin_t * sd
    return sin2, cos2


def _unit_uv(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-6):
    n = torch.sqrt(u * u + v * v + eps)
    return u / n, v / n


class PairVecContext(nn.Module):
    """
    把两路矢量 latent 变成一个“context”序列，用来条件化标量的两路融合。
    支持额外物理特征 extra: (B,L,N,P) 作为 gate 的条件信息。
    """
    def __init__(self, d_model: int, hidden: int = None, dropout_p: float = 0.05, extra_dim: int = 0):
        super().__init__()
        d_model = int(d_model)
        h = int(hidden) if hidden is not None else max(64, d_model)
        self.extra_dim = int(extra_dim)

        self.drop = nn.Dropout(p=float(dropout_p))
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model + self.extra_dim, h),
            nn.ReLU(),
            nn.Linear(h, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, va: torch.Tensor, vb: torch.Tensor, extra: torch.Tensor = None) -> torch.Tensor:
        # va,vb: (B,L,N,D)
        va = _nan_to_num(va, nan=0.0, posinf=0.0, neginf=0.0)
        vb = _nan_to_num(vb, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.cat([va, vb, va - vb, va * vb], dim=-1)  # (B,L,N,4D)

        if self.extra_dim > 0:
            if extra is None:
                # ablation: 如果你不想用物理特征，extra 自动补 0
                B, L, N, _ = x.shape
                extra = torch.zeros((B, L, N, self.extra_dim), device=x.device, dtype=x.dtype)
            else:
                extra = _nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
                if extra.shape[-1] != self.extra_dim:
                    raise RuntimeError(f"extra dim mismatch: got {extra.shape[-1]}, expected {self.extra_dim}")
            x = torch.cat([x, extra], dim=-1)

        x = self.drop(x)
        ctx = self.mlp(x)
        ctx = _nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        return self.norm(ctx)


class BiDAMFreqCtx(nn.Module):
    """
    你原来的实现（不改）
    """
    def __init__(self, d_model: int, gate_hidden: int = None, branch_drop_p: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        h = int(gate_hidden) if gate_hidden is not None else max(32, self.d_model // 4)

        self.gate = nn.Sequential(
            nn.Linear(4, h),
            nn.ReLU(),
            nn.Linear(h, 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.logit_temp = nn.Parameter(torch.tensor(0.5413248546))
        self.branch_drop_p = float(branch_drop_p)

        self.smooth = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if z_a.shape != z_b.shape or z_a.shape != ctx.shape:
            raise RuntimeError(f"BiDAMFreqCtx expects same shapes, got a={tuple(z_a.shape)} b={tuple(z_b.shape)} ctx={tuple(ctx.shape)}")
        if z_a.ndim != 4:
            raise RuntimeError(f"BiDAMFreqCtx expects (B,L,N,D), got {tuple(z_a.shape)}")

        B, L, N, D = z_a.shape
        if D != self.d_model:
            raise RuntimeError(f"d_model mismatch: got D={D}, expected {self.d_model}")

        z_a = _nan_to_num(z_a, nan=0.0, posinf=0.0, neginf=0.0)
        z_b = _nan_to_num(z_b, nan=0.0, posinf=0.0, neginf=0.0)
        ctx = _nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)

        a = z_a.permute(0, 2, 1, 3)  # (B,N,L,D)
        b = z_b.permute(0, 2, 1, 3)
        c = ctx.permute(0, 2, 1, 3)

        a_f = _nan_to_num_complex(torch.fft.rfft(a, dim=2))
        b_f = _nan_to_num_complex(torch.fft.rfft(b, dim=2))
        c_f = _nan_to_num_complex(torch.fft.rfft(c, dim=2))

        mag_a = torch.log1p(a_f.abs().mean(dim=-1))
        mag_b = torch.log1p(b_f.abs().mean(dim=-1))
        mag_c = torch.log1p(c_f.abs().mean(dim=-1))
        mag_a = _nan_to_num(mag_a, nan=0.0, posinf=20.0, neginf=0.0).clamp(0.0, 20.0)
        mag_b = _nan_to_num(mag_b, nan=0.0, posinf=20.0, neginf=0.0).clamp(0.0, 20.0)
        mag_c = _nan_to_num(mag_c, nan=0.0, posinf=20.0, neginf=0.0).clamp(0.0, 20.0)
        mag_m = (mag_a + mag_b) / 2.0

        gate_in = torch.stack([mag_a, mag_b, mag_c, mag_m], dim=-1)  # (B,N,F,4)
        gate_in = _nan_to_num(gate_in, nan=0.0, posinf=0.0, neginf=0.0)

        logits = self.gate(gate_in)  # (B,N,F,2)
        logits = _nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

        temp = F.softplus(self.logit_temp) + 1e-6
        w = _safe_softmax(logits / temp, dim=-1)  # (B,N,F,2)

        if self.training and self.branch_drop_p > 0.0:
            p = self.branch_drop_p
            trigger = (torch.rand((B, N), device=z_a.device) < p)
            if trigger.any():
                which = torch.randint(low=0, high=2, size=(B, N), device=z_a.device)
                mask = torch.ones((B, N, 1, 2), device=z_a.device, dtype=w.dtype)
                mask.scatter_(-1, which.view(B, N, 1, 1), 0.0)
                mask = torch.where(trigger.view(B, N, 1, 1), mask, torch.ones_like(mask))
                w = w * mask
                w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)

        w0 = w[..., 0:1]
        w1 = w[..., 1:2]

        sel_f = w0 * a_f + w1 * b_f
        mean_f = (a_f + b_f) / 2.0

        a_mix = torch.sigmoid(self.alpha)
        y_f = (1.0 - a_mix) * mean_f + a_mix * sel_f
        y_f = _nan_to_num_complex(y_f)

        y = torch.fft.irfft(y_f, n=L, dim=2)
        y = _nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        z = y.permute(0, 2, 1, 3)  # (B,L,N,D)

        z2 = z.permute(0, 2, 3, 1).reshape(B * N, D, L)
        z2 = self.smooth(z2).reshape(B, N, D, L).permute(0, 3, 1, 2)
        z2 = _nan_to_num(z2, nan=0.0, posinf=0.0, neginf=0.0)

        out = self.norm(z + z2)
        out = _nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class FNPFusionPhysPairwise(nn.Module):
    """
    你这份 pairwise 结构不变，只增强“矢量物理关联”：
      - SCADA vec 输入追加: cos/sin(风向-机舱)  （夹角信息）
      - ctx_obg / ctx_bgfc 追加物理对齐特征: cos/sin(矢量对齐)
      - 可选角度校准 delta_wdir/delta_nac
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
        obs_vec_feat: str = "uv+nac",
        # --- NEW: physics switches ---
        add_misalign_to_obsvec: bool = True,     # 把 cos/sin(Δψ) 加到 SCADA vec 输入
        add_align_to_ctx: bool = True,           # 把 cos/sin 对齐量加到 ctx
        use_angle_calib: bool = False,           # 学习 delta_wdir/delta_nac 解决方向定义偏差
        # ablation knobs
        obs_setconv_topk: int = None,
        obs_setconv_drop: float = 0.0,
        nfl_spectral_drop: float = 0.0,
        grid_point_drop: float = 0.0,
        grid_attn_drop: float = 0.0,
        bi_branch_drop_p: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(K_bg)
        self.fc_emb_scalar = int(fc_emb_scalar)
        self.fc_emb_vec = int(fc_emb_vec)

        self.obs_vec_feat = str(obs_vec_feat)
        self.add_misalign_to_obsvec = bool(add_misalign_to_obsvec)
        self.add_align_to_ctx = bool(add_align_to_ctx)
        self.use_angle_calib = bool(use_angle_calib)

        # 可学习角度校准（默认关闭）
        if self.use_angle_calib:
            self.delta_wdir = nn.Parameter(torch.zeros(()))
            self.delta_nac  = nn.Parameter(torch.zeros(()))

        # ===== encoders（不变）=====
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
            d_model=self.fc_emb_scalar, K=self.K, feat_per_point=1,
            use_setconv=bool(fc_use_setconv),
            point_drop=grid_point_drop, attn_drop=grid_attn_drop,
        )
        self.fc_scalar_pool = HorizonPool(e_dim=self.fc_emb_scalar, d_model=self.d_model)

        # ===== vector encoders =====
        if self.obs_vec_feat == "fl":
            base_dim = 2
        elif self.obs_vec_feat in ["uv+nac", "fl+nac"]:
            base_dim = 4
        else:
            raise ValueError(f"Unknown obs_vec_feat={self.obs_vec_feat}. Use 'fl'/'uv+nac'/'fl+nac'.")

        # NEW: 追加 misalignment (cos_mis,sin_mis)
        obs_vec_in_dim = base_dim + (2 if self.add_misalign_to_obsvec else 0)

        self.obs_vec_enc = FuncRepVFR(
            in_dim=obs_vec_in_dim, d_model=self.d_model, modes=modes,
            setconv_topk=obs_setconv_topk, setconv_drop=obs_setconv_drop,
            spectral_drop=nfl_spectral_drop
        )

        self.bg_vec_enc = GridEncoder(
            d_model=self.d_model, K=self.K, feat_per_point=2,
            use_setconv=bool(bg_use_setconv),
            point_drop=grid_point_drop, attn_drop=grid_attn_drop,
        )

        self.fc_vec_enc = GridEncoder(
            d_model=self.fc_emb_vec, K=self.K, feat_per_point=2,
            use_setconv=bool(fc_use_setconv),
            point_drop=grid_point_drop, attn_drop=grid_attn_drop,
        )
        self.fc_vec_pool = HorizonPool(e_dim=self.fc_emb_vec, d_model=self.d_model)

        # ===== vector -> scalar modulation =====
        self.obs_vec_film = VecFiLM(self.d_model, dropout_p=film_dropout_p)
        self.obs_film_scale = nn.Parameter(torch.tensor(-2.0))

        # ===== vector-context + pairwise fusion =====
        # ctx_bgfc 物理特征：cos/sin(an vs fc) -> 2
        # ctx_obg  物理特征：cos/sin(wind-nac) + cos/sin(sc vs an) -> 4
        extra_bgfc = 2 if self.add_align_to_ctx else 0
        extra_obg  = 4 if self.add_align_to_ctx else 0

        self.ctx_bgfc = PairVecContext(self.d_model, dropout_p=0.05, extra_dim=extra_bgfc)
        self.ctx_obg  = PairVecContext(self.d_model, dropout_p=0.05, extra_dim=extra_obg)

        self.fuse_bg_fc = BiDAMFreqCtx(self.d_model, branch_drop_p=bi_branch_drop_p)
        self.fuse_obs_bgfc = BiDAMFreqCtx(self.d_model, branch_drop_p=bi_branch_drop_p)

        self.out_norm = nn.LayerNorm(self.d_model)
        self._warned_nonfinite = False

    def _build_obs_vec(self, W3, dir_sin, dir_cos, nac_sin, nac_cos, cos_mis=None, sin_mis=None):
        u_sc, v_sc = _wind_uv_from_speed_dir_sincos(W3, dir_sin, dir_cos)

        if self.obs_vec_feat == "fl":
            f_sc, l_sc = _rotate_uv_to_nacelle_frame(u_sc, v_sc, nac_sin, nac_cos)
            base = torch.stack([f_sc, l_sc], dim=-1)  # (B,L,N,2)
        elif self.obs_vec_feat == "uv+nac":
            base = torch.stack([u_sc, v_sc, nac_sin, nac_cos], dim=-1)  # (B,L,N,4)
        elif self.obs_vec_feat == "fl+nac":
            f_sc, l_sc = _rotate_uv_to_nacelle_frame(u_sc, v_sc, nac_sin, nac_cos)
            base = torch.stack([f_sc, l_sc, nac_sin, nac_cos], dim=-1)  # (B,L,N,4)
        else:
            raise RuntimeError("unreachable")

        if self.add_misalign_to_obsvec:
            if cos_mis is None or sin_mis is None:
                cos_mis = dir_cos * nac_cos + dir_sin * nac_sin
                sin_mis = dir_sin * nac_cos - dir_cos * nac_sin
            extra = torch.stack([cos_mis, sin_mis], dim=-1)  # (B,L,N,2)
            base = torch.cat([base, extra], dim=-1)

        return base

    def forward(self, x_obs, x_an, coords, pos, fc0, x_an_valid=None, fc0v=None):
        if (not self._warned_nonfinite):
            for name, t in [("x_obs", x_obs), ("x_an", x_an), ("coords", coords), ("pos", pos), ("fc0", fc0)]:
                if isinstance(t, torch.Tensor) and (not torch.isfinite(t).all()):
                    print(f"[FNPFusionPhysPairwise] WARNING: non-finite detected in {name}. nan_to_num safeguards are applied.")
                    self._warned_nonfinite = True
                    break

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
            raise RuntimeError(f"fc0 shape mismatch, expected (B,H,N,K,4) with N={N},K={self.K}, got {tuple(fc0.shape)}")

        dyn_an = x_an.view(B, L, N, self.K, 4)
        dyn_an_v = x_an_valid.view(B, L, N, self.K, 4) if x_an_valid is not None else None

        # =========================
        # 1) scalar enc
        # =========================
        x_obs_scalar = x_obs[..., 0:3]  # [P3,dP3,W3]

        an_speed = dyn_an[..., 0:1]
        an_speed_v = dyn_an_v[..., 0:1] if dyn_an_v is not None else None

        fc0_speed = fc0[..., 0:1]
        fc0_speed_v = fc0v[..., 0:1] if fc0v is not None else None

        z_obs_s = self.obs_scalar_enc(x_obs_scalar, coords)           # (B,L,N,D)
        z_bg_s  = self.bg_scalar_enc(an_speed, pos, valid=an_speed_v) # (B,L,N,D)

        e_fc_s  = self.fc_scalar_enc(fc0_speed, pos, valid=fc0_speed_v)  # (B,H,N,Es)
        z_fc_s0 = self.fc_scalar_pool(e_fc_s)                            # (B,N,D)
        z_fc_s  = z_fc_s0.unsqueeze(1).expand(B, L, N, self.d_model)      # (B,L,N,D)

        # =========================
        # 2) vector enc + NEW physics features
        # =========================
        W3      = x_obs[..., 2]
        dir_sin = x_obs[..., 3]
        dir_cos = x_obs[..., 4]
        nac_sin = x_obs[..., 5]
        nac_cos = x_obs[..., 6]

        # 可选：角度小校准（解决数据集间方向定义差异）
        if self.use_angle_calib:
            dir_sin, dir_cos = _rotate_sincos_by_delta(dir_sin, dir_cos, self.delta_wdir)
            nac_sin, nac_cos = _rotate_sincos_by_delta(nac_sin, nac_cos, self.delta_nac)

        # misalignment: wind_dir - nac
        cos_mis = dir_cos * nac_cos + dir_sin * nac_sin
        sin_mis = dir_sin * nac_cos - dir_cos * nac_sin

        # SCADA earth-frame u,v（用于对齐计算）
        u_sc, v_sc = _wind_uv_from_speed_dir_sincos(W3, dir_sin, dir_cos)
        sc_u_hat, sc_v_hat = _unit_uv(u_sc, v_sc)

        # SCADA vector features（可能在 nac frame / 或带 nac）
        x_obs_vec = self._build_obs_vec(W3, dir_sin, dir_cos, nac_sin, nac_cos, cos_mis=cos_mis, sin_mis=sin_mis)
        z_obs_v = self.obs_vec_enc(x_obs_vec, coords)  # (B,L,N,D)

        # analysis vectors (earth frame u,v)
        an_vec = dyn_an[..., 2:4]  # (B,L,N,K,2) [u,v]
        an_uv_v = dyn_an_v[..., 2:4] if dyn_an_v is not None else None
        z_bg_v = self.bg_vec_enc(an_vec, pos, valid=an_uv_v)  # (B,L,N,D)

        # forecast vectors (earth frame u,v)
        fc_vec = fc0[..., 2:4]  # (B,H,N,K,2)
        fc_uv_v = fc0v[..., 2:4] if fc0v is not None else None
        e_fc_v  = self.fc_vec_enc(fc_vec, pos, valid=fc_uv_v)  # (B,H,N,Ev)
        z_fc_v0 = self.fc_vec_pool(e_fc_v)                      # (B,N,D)
        z_fc_v  = z_fc_v0.unsqueeze(1).expand(B, L, N, self.d_model)  # (B,L,N,D)

        # =========================
        # 3) vector -> modulate SCADA scalar
        # =========================
        gamma, beta = self.obs_vec_film(z_obs_v)
        s = torch.sigmoid(self.obs_film_scale)
        z_obs_s = (1.0 + s * gamma) * z_obs_s + (s * beta)

        # =========================
        # 4) build physics-conditioned ctx extras
        # =========================
        extra_bgfc = None
        extra_obg = None
        if self.add_align_to_ctx:
            # --- SCADA vs analysis alignment (time-dependent) ---
            u_an_mean = an_vec[..., 0].mean(dim=3)  # (B,L,N)
            v_an_mean = an_vec[..., 1].mean(dim=3)
            an_u_hat, an_v_hat = _unit_uv(u_an_mean, v_an_mean)

            cos_sa = sc_u_hat * an_u_hat + sc_v_hat * an_v_hat  # (B,L,N)
            sin_sa = sc_u_hat * an_v_hat - sc_v_hat * an_u_hat  # (B,L,N)

            # ctx_obg extra: [cos_mis,sin_mis, cos_sa,sin_sa]
            extra_obg = torch.stack([cos_mis, sin_mis, cos_sa, sin_sa], dim=-1)  # (B,L,N,4)

            # --- analysis(t0) vs forecast(mean) alignment (time-constant over L) ---
            an0_u = an_u_hat[:, -1]  # (B,N)
            an0_v = an_v_hat[:, -1]

            u_fc_mean = fc_vec[..., 0].mean(dim=3)  # (B,H,N)
            v_fc_mean = fc_vec[..., 1].mean(dim=3)
            fc_u_hat_h, fc_v_hat_h = _unit_uv(u_fc_mean, v_fc_mean)  # (B,H,N)
            fc_u_bar = fc_u_hat_h.mean(dim=1)  # (B,N)
            fc_v_bar = fc_v_hat_h.mean(dim=1)
            fc_u_bar, fc_v_bar = _unit_uv(fc_u_bar, fc_v_bar)

            cos_af = an0_u * fc_u_bar + an0_v * fc_v_bar  # (B,N)
            sin_af = an0_u * fc_v_bar - an0_v * fc_u_bar  # (B,N)

            extra_bgfc = torch.stack([cos_af, sin_af], dim=-1).unsqueeze(1).expand(B, L, N, 2)  # (B,L,N,2)

        # =========================
        # 5) pairwise scalar fusion conditioned by vector contexts (+ physics extras)
        # =========================
        ctx_bgfc = self.ctx_bgfc(z_bg_v, z_fc_v, extra=extra_bgfc)
        z_bgfc_s = self.fuse_bg_fc(z_bg_s, z_fc_s, ctx_bgfc)

        ctx_obg = self.ctx_obg(z_obs_v, z_bg_v, extra=extra_obg)
        z = self.fuse_obs_bgfc(z_obs_s, z_bgfc_s, ctx_obg)

        z = self.out_norm(z)
        z = _nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        e_fc = torch.cat([e_fc_s, e_fc_v], dim=-1)  # (B,H,N,Es+Ev)
        return z, e_fc
