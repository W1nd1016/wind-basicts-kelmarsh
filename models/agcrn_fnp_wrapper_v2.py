# models/agcrn_fnp_wrapper_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BgFutureCorrectionHead(nn.Module):
    """
    z_bg_future: (B,H,N,D) -> delta: (B,H,N)
    """
    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_bg_future):
        return self.mlp(z_bg_future).squeeze(-1)

class FNP_AGCRN_ModelV3(nn.Module):
    """
    z_hist(FNP) -> AGCRN -> base
    z_bg_future(FNP) -> correction -> final = base + softplus(gamma)*delta
    """
    def __init__(self, fnp_fusion, agcrn_model, d_model: int):
        super().__init__()
        self.fnp = fnp_fusion
        self.agcrn = agcrn_model
        self.corr = BgFutureCorrectionHead(d_model=d_model, hidden=64)
        self.gamma = nn.Parameter(torch.tensor(-2.0))  # start small after softplus

    def forward(
        self,
        x_obs, x_bg, coords, dx, dy, ds,
        x_obs_v=None, x_bg_v=None,
        teacher_forcing_y=None,
        teacher_forcing_ratio: float = 0.0,
    ):
        z_hist, z_bg_future = self.fnp(
            x_obs, x_bg, coords, dx, dy, ds,
            x_obs_v=x_obs_v, x_bg_v=x_bg_v
        )

        if teacher_forcing_y is None:
            base = self.agcrn(z_hist)
        else:
            base = self.agcrn(z_hist, teacher_forcing_y=teacher_forcing_y, teacher_forcing_ratio=teacher_forcing_ratio)

        delta = self.corr(z_bg_future)
        return base + F.softplus(self.gamma) * delta
