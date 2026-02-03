# models/agcrn_s2_wrapper_FnpOnlyAnalysis_ab_no_forecast.py
import torch
import torch.nn as nn


class S2Model(nn.Module):
    def __init__(self, fnp_fusion, agcrn_model, d_model: int):
        super().__init__()
        self.fnp = fnp_fusion
        self.agcrn = agcrn_model
        self.z_to_x = nn.Linear(int(d_model), int(d_model))

    def forward(
        self,
        x_obs,
        x_an,
        coords,
        fc0,
        pos,
        x_an_valid=None,
        fc0v=None,
        teacher_forcing_y=None,
        teacher_forcing_ratio: float = 0.0,
    ):
        z, e_fc = self.fnp(
            x_obs=x_obs,
            x_an=x_an,
            coords=coords,
            pos=pos,
            fc0=fc0,
            x_an_valid=x_an_valid,
            fc0v=fc0v,
        )

        x = self.z_to_x(z)

        y_hat = self.agcrn(
            x,
            exog=e_fc,  # e_fc is None when exog_dim=0
            teacher_forcing_y=teacher_forcing_y,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return y_hat
