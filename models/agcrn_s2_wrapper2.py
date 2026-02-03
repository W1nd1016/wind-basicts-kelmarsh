# models/agcrn_s2_wrapper.py
import torch
import torch.nn as nn

class ForecastCorrectionHead(nn.Module):
    """
    fc0: (B,H,N,F_fc) -> delta: (B,H,N)
    """
    def __init__(self, f_fc: int, hidden: int = 64):
        super().__init__()
        self.f_fc = int(f_fc)
        self.mlp = nn.Sequential(
            nn.Linear(self.f_fc, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, fc0: torch.Tensor) -> torch.Tensor:
        # fc0: (B,H,N,F_fc)
        if fc0.ndim != 4:
            raise RuntimeError(f"ForecastCorrectionHead expects fc0 (B,H,N,F_fc), got {tuple(fc0.shape)}")
        if fc0.shape[-1] != self.f_fc:
            raise RuntimeError(f"f_fc mismatch: fc0.F={fc0.shape[-1]}, expected {self.f_fc}")
        return self.mlp(fc0).squeeze(-1)  # (B,H,N)

class S2Model(nn.Module):
    """
    S2 model:
      z(FNP) -> decoder(MLP) -> base
      fc0 -> correction -> final = base + gamma * delta

    NOTE:
      - decoder should accept (B,L,N,D) and output (B,H,N)
      - teacher_forcing args are kept for compatibility, but ignored for MLP decoder
    """
    def __init__(self, fnp_fusion: nn.Module, decoder: nn.Module, d_model: int, f_fc: int):
        super().__init__()
        self.fnp = fnp_fusion
        self.decoder = decoder
        self.z_to_x = nn.Linear(int(d_model), int(d_model))

        self.f_fc = int(f_fc)
        self.corr = ForecastCorrectionHead(self.f_fc) if self.f_fc > 0 else None
        self.gamma = nn.Parameter(torch.tensor(0.0))  # start from 0

    def forward(
        self,
        x_obs: torch.Tensor,
        x_bg: torch.Tensor,
        coords: torch.Tensor,
        fc0: torch.Tensor,
        x_bg_valid: torch.Tensor = None,
        fc0v: torch.Tensor = None,
        teacher_forcing_y: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        # FNP frontend
        z = self.fnp(x_obs, x_bg, coords, x_bg_valid=x_bg_valid)  # (B,L,N,D)
        x = self.z_to_x(z)

        # Base forecast from decoder (MLP)
        base = self.decoder(x)  # (B,H,N)

        # Optional forecast correction
        if (self.corr is None) or (fc0 is None) or (fc0.numel() == 0):
            return base

        delta = self.corr(fc0)  # (B,H,N)

        if fc0v is not None:
            # fc0v: (B,H,N,F_fc) -> gate (B,H,N)
            gate = fc0v.float().mean(dim=-1)
            delta = delta * gate

        return base + self.gamma * delta
