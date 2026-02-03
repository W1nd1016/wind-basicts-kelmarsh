# models/fnp_fusion_vecbranch_FnpOnlyAnalysis.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fnp_fusion_OnlyAnalysis import (
    FuncRepVFR,
    BgGridEncoderAnOnly,
    FcGridEncoder,
    FcHorizonPool,
    TriDAMSoft,
)
from models.vector_branch_nacelle import VectorSetEncoder2D, angle_to_sincos


class FNPFusionVecBranch(nn.Module):
    """
    Scalar/Vector separated frontend with nacelle-frame vector branch.

    Inputs (normalized):
      x_obs: (B,L,N,7)   [P3,dP3,W3, dir_sin,dir_cos, nac_sin,nac_cos]  (your current layout)
      x_an : (B,L,N,K*4) analysis (speed, direction, u, v) per neighbor point
      fc0  : (B,H,N,K,4) forecast@t0 (speed, direction, u, v)

    We DO:
      - Scalar branches:
          obs_scalar  = [P3,dP3,W3] -> FuncRepVFR
          an_scalar   = speed only  -> BgGridEncoderAnOnly(feat_per_point=1)
          fc_scalar   = speed only  -> FcGridEncoder(feat_per_point=1) + pool -> context
          z_scalar = TriDAMSoft(z_obs, z_bg, z_fcctx)
      - Vector branches (nacelle-frame, using direction only):
          v_scada(t)  = W3_raw * [cos(dir-nac), sin(dir-nac)]
          v_an(t,k)   = speed_raw * [cos(dir_k-nac), sin(dir_k-nac)]
          v_fc(h,k)   = speed_raw * [cos(dir_fc-nac_t0), sin(dir_fc-nac_t0)]
        Encode with VectorSetEncoder2D and:
          z = FiLM(z_scalar; z_vec_hist)
          e_fc = fuse([e_fc_scalar, e_fc_vec]) -> keep exog_dim = fc_emb_dim

    Returns:
      z   : (B,L,N,D)
      e_fc: (B,H,N,fc_emb_dim)
    """
    def __init__(
        self,
        d_model: int,
        fc_emb_dim: int,
        K_bg: int,
        modes: int = 8,
        bg_use_setconv: bool = True,
        fc_use_setconv: bool = True,
        x_mu: torch.Tensor | None = None,
        x_sd: torch.Tensor | None = None,
        fc_mu: torch.Tensor | None = None,
        fc_sd: torch.Tensor | None = None,
        vec_hidden: int = 128,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.fc_emb_dim = int(fc_emb_dim)
        self.K = int(K_bg)

        # --- scalar branches ---
        self.obs_enc = FuncRepVFR(in_dim=3, d_model=self.d_model, modes=int(modes))
        self.bg_enc = BgGridEncoderAnOnly(
            d_model=self.d_model,
            K=self.K,
            feat_per_point=1,
            use_setconv=bool(bg_use_setconv),
        )
        self.fc_enc = FcGridEncoder(
            e_dim=self.fc_emb_dim,
            K=self.K,
            feat_per_point=1,
            use_setconv=bool(fc_use_setconv),
        )
        self.fc_pool = FcHorizonPool(e_dim=self.fc_emb_dim, d_model=self.d_model)
        self.tri_dam = TriDAMSoft(self.d_model)

        # --- vector encoders ---
        self.vec_hist_enc = VectorSetEncoder2D(out_dim=self.d_model, hidden=int(vec_hidden), use_pos=True)
        self.vec_fc_enc   = VectorSetEncoder2D(out_dim=self.fc_emb_dim, hidden=int(vec_hidden), use_pos=True)

        # FiLM modulation from vector -> (gamma,beta)
        self.vec_to_film = nn.Linear(self.d_model, 2 * self.d_model)
        self.out_norm = nn.LayerNorm(self.d_model)

        # fuse scalar forecast embedding + vector forecast embedding -> exog_dim stays fc_emb_dim
        self.fc_fuse = nn.Linear(2 * self.fc_emb_dim, self.fc_emb_dim)
        self.fc_norm = nn.LayerNorm(self.fc_emb_dim)

        # stats buffers (for de-normalizing only the things we need)
        if (x_mu is None) or (x_sd is None):
            raise RuntimeError("FNPFusionVecBranch requires x_mu and x_sd from meta.json")
        if (fc_mu is None) or (fc_sd is None):
            raise RuntimeError("FNPFusionVecBranch requires fc_mu and fc_sd from meta.json")

        x_mu = x_mu.float()
        x_sd = x_sd.float()
        self.register_buffer("x_mu", x_mu)
        self.register_buffer("x_sd", x_sd)

        fc_mu = fc_mu.float()
        fc_sd = fc_sd.float()
        self.register_buffer("fc_mu", fc_mu)
        self.register_buffer("fc_sd", fc_sd)

        # sanity (expected X features: 7 + K*4)
        if self.x_mu.numel() != 7 + self.K * 4:
            raise RuntimeError(f"x_mu length mismatch: got {self.x_mu.numel()}, expected {7 + self.K*4}")
        if self.fc_mu.ndim != 3 or self.fc_mu.size(-1) != 4 or self.fc_mu.size(1) != self.K:
            raise RuntimeError(f"fc_mu shape mismatch: got {tuple(self.fc_mu.shape)}, expected (H,K,4) with K={self.K}")

        # pre-sliced stats for obs/an
        self.register_buffer("obs_mu", self.x_mu[:7].view(1, 1, 1, 7))
        self.register_buffer("obs_sd", self.x_sd[:7].view(1, 1, 1, 7))
        self.register_buffer("an_mu", self.x_mu[7:].view(1, 1, 1, self.K, 4))
        self.register_buffer("an_sd", self.x_sd[7:].view(1, 1, 1, self.K, 4))

    def _denorm_obs(self, x_obs: torch.Tensor) -> torch.Tensor:
        # x_obs: (B,L,N,7)
        return x_obs * self.obs_sd + self.obs_mu

    def _denorm_an(self, x_an: torch.Tensor) -> torch.Tensor:
        # x_an: (B,L,N,K,4)
        return x_an * self.an_sd + self.an_mu

    def _denorm_fc(self, fc0: torch.Tensor) -> torch.Tensor:
        # fc0: (B,H,N,K,4)
        # fc_mu/sd are (H,K,4) -> broadcast to (B,H,N,K,4)
        mu = self.fc_mu.unsqueeze(0).unsqueeze(2)  # (1,H,1,K,4)
        sd = self.fc_sd.unsqueeze(0).unsqueeze(2)  # (1,H,1,K,4)
        return fc0 * sd + mu

    @staticmethod
    def _rel_sin_cos_from_sincos(dir_sin: torch.Tensor, dir_cos: torch.Tensor, nac_sin: torch.Tensor, nac_cos: torch.Tensor):
        # sin(a-b) = sin a cos b - cos a sin b
        # cos(a-b) = cos a cos b + sin a sin b
        sin_rel = dir_sin * nac_cos - dir_cos * nac_sin
        cos_rel = dir_cos * nac_cos + dir_sin * nac_sin
        return sin_rel, cos_rel

    def forward(
        self,
        x_obs: torch.Tensor,        # (B,L,N,7) normalized
        x_obs_valid: torch.Tensor,  # (B,L,N,7) 0/1
        x_an: torch.Tensor,         # (B,L,N,K*4) normalized
        coords: torch.Tensor,       # (L*N,3)
        fc0: torch.Tensor,          # (B,H,N,K,4) normalized
        pos: torch.Tensor,          # (N,K,3)
        x_an_valid: torch.Tensor | None = None,  # (B,L,N,K*4) 0/1
        fc0v: torch.Tensor | None = None,        # (B,H,N,K,4) 0/1
    ):
        if x_obs.ndim != 4 or x_obs.size(-1) != 7:
            raise RuntimeError(f"x_obs must be (B,L,N,7), got {tuple(x_obs.shape)}")
        if x_an.ndim != 4 or x_an.size(-1) != self.K * 4:
            raise RuntimeError(f"x_an must be (B,L,N,K*4) with K={self.K}, got {tuple(x_an.shape)}")
        if fc0.ndim != 5 or fc0.size(3) != self.K or fc0.size(-1) != 4:
            raise RuntimeError(f"fc0 must be (B,H,N,K,4) with K={self.K}, got {tuple(fc0.shape)}")

        B, L, N, _ = x_obs.shape
        H = fc0.size(1)

        # ---------- scalar branches ----------
        x_obs_scalar = x_obs[..., :3]  # (B,L,N,3)
        z_obs = self.obs_enc(x_obs_scalar, coords)  # (B,L,N,D)

        an4 = x_an.view(B, L, N, self.K, 4)
        if x_an_valid is not None:
            anv4 = x_an_valid.view(B, L, N, self.K, 4)
        else:
            anv4 = torch.ones_like(an4)

        an_speed = an4[..., 0:1]       # (B,L,N,K,1) (normalized)
        an_speed_v = anv4[..., 0:1]    # (B,L,N,K,1)
        z_bg = self.bg_enc(an_speed, pos, x_an_valid=an_speed_v)  # (B,L,N,D)

        fc_speed = fc0[..., 0:1]       # (B,H,N,K,1)
        if fc0v is not None:
            fc_speed_v = fc0v[..., 0:1]
        else:
            fc_speed_v = torch.ones_like(fc_speed)

        e_fc_s = self.fc_enc(fc_speed, pos, fc0v=fc_speed_v)  # (B,H,N,E)
        z_fc0 = self.fc_pool(e_fc_s)                           # (B,N,D)
        z_fc = z_fc0.unsqueeze(1).expand(B, L, N, self.d_model)
        z_scalar = self.tri_dam(z_obs=z_obs, z_bg=z_bg, z_fc=z_fc)  # (B,L,N,D)

        # ---------- vector branches (nacelle-frame) ----------
        # de-normalize only what we need
        x_obs_raw = self._denorm_obs(x_obs)  # (B,L,N,7)

        W3 = x_obs_raw[..., 2]   # (B,L,N) raw wind speed
        dir_sin = x_obs_raw[..., 3]
        dir_cos = x_obs_raw[..., 4]
        nac_sin = x_obs_raw[..., 5]
        nac_cos = x_obs_raw[..., 6]

        # scada valid (need W, dir_sin/cos, nac_sin/cos)
        v_scada_valid = (
            (x_obs_valid[..., 2] > 0.5)
            & (x_obs_valid[..., 3] > 0.5)
            & (x_obs_valid[..., 4] > 0.5)
            & (x_obs_valid[..., 5] > 0.5)
            & (x_obs_valid[..., 6] > 0.5)
        ).float()  # (B,L,N)

        sin_rel_sc, cos_rel_sc = self._rel_sin_cos_from_sincos(dir_sin, dir_cos, nac_sin, nac_cos)
        v_scada = torch.stack([W3 * cos_rel_sc, W3 * sin_rel_sc], dim=-1)  # (B,L,N,2)

        # CERRA analysis: use speed + direction (raw) -> nacelle-frame vector
        an_raw = self._denorm_an(an4)       # (B,L,N,K,4) raw
        an_speed_raw = an_raw[..., 0]       # (B,L,N,K)
        an_dir_raw = an_raw[..., 1]         # (B,L,N,K)

        an_dir_sin, an_dir_cos = angle_to_sincos(an_dir_raw)  # auto-deg/rad
        nac_sin_k = nac_sin.unsqueeze(3)  # (B,L,N,1)
        nac_cos_k = nac_cos.unsqueeze(3)

        sin_rel_an = an_dir_sin * nac_cos_k - an_dir_cos * nac_sin_k
        cos_rel_an = an_dir_cos * nac_cos_k + an_dir_sin * nac_sin_k

        v_an = torch.stack([an_speed_raw * cos_rel_an, an_speed_raw * sin_rel_an], dim=-1)  # (B,L,N,K,2)

        if x_an_valid is not None:
            v_an_valid = ((anv4[..., 0] > 0.5) & (anv4[..., 1] > 0.5)).float()  # (B,L,N,K)
        else:
            v_an_valid = torch.ones((B, L, N, self.K), device=x_obs.device, dtype=x_obs.dtype)

        # merge scada vector as k=0 with pos=(0,0,0)
        v0 = v_scada.unsqueeze(3)            # (B,L,N,1,2)
        m0 = v_scada_valid.unsqueeze(3)      # (B,L,N,1)
        v_hist = torch.cat([v0, v_an], dim=3)     # (B,L,N,K+1,2)
        m_hist = torch.cat([m0, v_an_valid], dim=3)  # (B,L,N,K+1)

        pos0 = torch.zeros((pos.size(0), 1, 3), device=pos.device, dtype=pos.dtype)  # (N,1,3)
        pos_aug = torch.cat([pos0, pos], dim=1)  # (N,K+1,3)

        z_vec = self.vec_hist_enc(v_hist, pos_aug, m=m_hist)  # (B,L,N,D)

        # FiLM modulation
        film = self.vec_to_film(z_vec)  # (B,L,N,2D)
        gamma, beta = film[..., :self.d_model], film[..., self.d_model:]
        z = self.out_norm((1.0 + gamma) * z_scalar + beta)  # (B,L,N,D)

        # ---------- forecast vector embedding ----------
        # use nacelle at t0 = last history step
        nac_sin0 = nac_sin[:, -1]  # (B,N)
        nac_cos0 = nac_cos[:, -1]  # (B,N)

        fc_raw = self._denorm_fc(fc0)        # (B,H,N,K,4) raw
        fc_speed_raw = fc_raw[..., 0]        # (B,H,N,K)
        fc_dir_raw = fc_raw[..., 1]          # (B,H,N,K)

        fc_dir_sin, fc_dir_cos = angle_to_sincos(fc_dir_raw)

        nac_sin0_k = nac_sin0.unsqueeze(1).unsqueeze(3)  # (B,1,N,1)
        nac_cos0_k = nac_cos0.unsqueeze(1).unsqueeze(3)

        sin_rel_fc = fc_dir_sin * nac_cos0_k - fc_dir_cos * nac_sin0_k
        cos_rel_fc = fc_dir_cos * nac_cos0_k + fc_dir_sin * nac_sin0_k

        v_fc = torch.stack([fc_speed_raw * cos_rel_fc, fc_speed_raw * sin_rel_fc], dim=-1)  # (B,H,N,K,2)

        if fc0v is not None:
            m_fc = ((fc0v[..., 0] > 0.5) & (fc0v[..., 1] > 0.5)).float()  # speed & direction valid
        else:
            m_fc = torch.ones((B, H, N, self.K), device=x_obs.device, dtype=x_obs.dtype)

        e_fc_v = self.vec_fc_enc(v_fc, pos, m=m_fc)  # (B,H,N,E)

        # fuse exog: keep exog_dim == fc_emb_dim
        e_fc = self.fc_fuse(torch.cat([e_fc_s, e_fc_v], dim=-1))
        e_fc = self.fc_norm(e_fc)
        return z, e_fc
