# models/agcrn_s2_wrapper_FnpOnlyAnalysis.py
import torch
import torch.nn as nn


class S2Model(nn.Module):
    """
    S2 with FNP (forecast scalar only, vector branch removed):

      FNPFusion returns:
        z:    (B, L, N, D)
        e_fc: (B, H, N, E_fc)   # scalar forecast embedding only

      AGCRN decoder uses exog = e_fc.

    Output:
      y_hat: (B, H, N)
    """
    def __init__(
        self,
        fnp_fusion,
        agcrn_model,
        d_model: int,
    ):
        super().__init__()
        self.fnp = fnp_fusion
        self.agcrn = agcrn_model
        self.z_to_x = nn.Linear(int(d_model), int(d_model))

    def forward(
        self,
        x_obs,          # (B,L,N,7)
        x_an,           # (B,L,N,K*4)
        coords,         # (L*N,3)
        fc0,            # (B,H,N,K,4)   forecast scalar inputs
        pos,            # (N,K,3) or (B,N,K,3)
        x_an_valid=None,
        fc0v=None,      # kept for interface compatibility (unused)
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
            fc0v=fc0v,   # ignored inside fusion
        )  # z:(B,L,N,D), e_fc:(B,H,N,E_fc)

        x = self.z_to_x(z)

        y_hat = self.agcrn(
            x,
            exog=e_fc,
            teacher_forcing_y=teacher_forcing_y,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return y_hat
