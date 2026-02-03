# models/s2_mlp_decoder.py
import torch
import torch.nn as nn

class MLPSeqDecoder(nn.Module):
    """
    Standard MLP decoder for multi-step forecasting.

    Input:
      z: (B, L, N, D)  frontend features (e.g., FNP output)
    Output:
      y: (B, H, N)

    pooling:
      - "flatten": flatten all history (L*D) per node as input to MLP (standard, strong baseline)
      - "last":    use last step (D) per node as input to MLP (simpler)
      - "mean":    mean over time (D) per node as input to MLP
    """
    def __init__(
        self,
        d_model: int,
        L: int,
        H: int,
        hidden: int = 256,
        dropout: float = 0.1,
        pooling: str = "flatten",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.L = int(L)
        self.H = int(H)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.pooling = str(pooling).lower()

        if self.pooling not in {"flatten", "last", "mean"}:
            raise ValueError(f"Invalid pooling='{pooling}'. Use one of: flatten/last/mean.")

        if self.pooling == "flatten":
            in_dim = self.L * self.d_model
        else:
            in_dim = self.d_model

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.H),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,L,N,D)
        if z.ndim != 4:
            raise RuntimeError(f"MLPSeqDecoder expects z with shape (B,L,N,D), but got {tuple(z.shape)}")

        B, L, N, D = z.shape
        if D != self.d_model:
            raise RuntimeError(f"d_model mismatch: z.D={D}, expected {self.d_model}")
        if self.pooling == "flatten" and L != self.L:
            raise RuntimeError(f"L mismatch for flatten pooling: z.L={L}, expected {self.L}")

        if self.pooling == "flatten":
            # (B,N,L,D) -> (B,N,L*D)
            feat = z.permute(0, 2, 1, 3).contiguous().view(B, N, L * D)
        elif self.pooling == "last":
            # last step: (B,N,D)
            feat = z[:, -1].permute(0, 1, 2).contiguous()  # (B,N,D)
            feat = feat  # keep as (B,N,D)
        else:  # "mean"
            feat = z.mean(dim=1)  # (B,N,D)

        # MLP node-wise
        out = self.mlp(feat)  # (B,N,H)

        # -> (B,H,N)
        out = out.permute(0, 2, 1).contiguous()
        return out
