import torch
import torch.nn as nn


class ForecastCorrectionHead(nn.Module):
    """
    Forecast correction head:
      fc0: (B,H,N,F_fc) -> delta: (B,H,N)

    It predicts a scalar correction per (time,node) from forecast feature vector.
    """
    def __init__(self, f_fc, hidden=64):
        super().__init__()
        self.f_fc = int(f_fc)
        self.mlp = nn.Sequential(
            nn.Linear(self.f_fc, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, fc0):
        return self.mlp(fc0).squeeze(-1)  # (B,H,N)


class S2Model(nn.Module):
    """
    S2 model:
      z = FNPFusion(x_obs, x_bg, coords)        -> (B,L,N,D)
      x = Linear(z)                            -> (B,L,N,D)
      base = AGCRNSeq2SeqBaseline(x)           -> (B,H,N)
      delta = CorrectionHead(fc0)              -> (B,H,N)
      pred = base + gamma * delta

    Note:
      With B1 change, AGCRNSeq2SeqBaseline now generates its decoder seed y_prev
      from the FULL latent vector X_last (B,N,D) via an internal seed MLP.
    """
    def __init__(self, fnp_fusion, agcrn_model, d_model, f_fc):
        super().__init__()
        self.fnp = fnp_fusion
        self.agcrn = agcrn_model

        self.z_to_x = nn.Linear(int(d_model), int(d_model))

        self.f_fc = int(f_fc)
        self.corr = ForecastCorrectionHead(self.f_fc) if self.f_fc > 0 else None
        self.gamma = nn.Parameter(torch.tensor(0.0))  # start from 0

    def forward(
        self,
        x_obs, x_bg, coords, fc0,
        x_bg_valid=None, fc0v=None,
        teacher_forcing_y=None, teacher_forcing_ratio=0.0
    ):
        # FNP fusion: obs+bg -> latent z
        z = self.fnp(x_obs, x_bg, coords, x_bg_valid=x_bg_valid)  # (B,L,N,D)
        x = self.z_to_x(z)                                        # (B,L,N,D)

        # AGCRN seq2seq (autoregressive decoder inside)
        if teacher_forcing_y is None:
            base = self.agcrn(x)  # (B,H,N)
        else:
            base = self.agcrn(x, teacher_forcing_y=teacher_forcing_y, teacher_forcing_ratio=teacher_forcing_ratio)

        # optional correction
        if (self.corr is None) or (fc0 is None) or (fc0.numel() == 0):
            return base

        delta = self.corr(fc0)  # (B,H,N)

        if fc0v is not None:
            # gate by forecast validity: average over feature dim
            gate = fc0v.float().mean(dim=-1)  # (B,H,N)
            delta = delta * gate

        return base + self.gamma * delta
