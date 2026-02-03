# models/dcrnn_seq2seq_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionConv(nn.Module):
    """
    Diffusion convolution from DCRNN:
    Given supports {S_i}, compute [X, S_i X, S_i^2 X, ...] then linear project.
    """
    def __init__(self, in_dim: int, out_dim: int, K: int, num_supports: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.K = int(K)
        self.num_supports = int(num_supports)

        fan_in = self.in_dim * (1 + self.K * self.num_supports)
        self.proj = nn.Linear(fan_in, self.out_dim)

    def forward(self, x: torch.Tensor, supports: list):
        """
        x: (B,N,Cin)
        supports: list of (N,N) tensors
        """
        B, N, Cin = x.shape
        outs = [x]

        for S in supports:
            xk = x
            for _ in range(self.K):
                # (N,N) x (B,N,C) -> (B,N,C)
                xk = torch.einsum("nm,bmc->bnc", S, xk)
                outs.append(xk)

        x_cat = torch.cat(outs, dim=-1)  # (B,N,Cin*(1+K*num_supports))
        return self.proj(x_cat)


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell (DCGRU)
    """
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, K: int, num_supports: int, dropout: float = 0.0):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.num_supports = int(num_supports)

        self.conv_gates = DiffusionConv(self.input_dim + self.hidden_dim, 2 * self.hidden_dim, K=self.K, num_supports=self.num_supports)
        self.conv_cand  = DiffusionConv(self.input_dim + self.hidden_dim, self.hidden_dim, K=self.K, num_supports=self.num_supports)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, supports: list):
        """
        x_t: (B,N,input_dim)
        h_prev: (B,N,hidden_dim)
        """
        if h_prev is None:
            h_prev = torch.zeros((x_t.size(0), x_t.size(1), self.hidden_dim), device=x_t.device, dtype=x_t.dtype)

        xh = torch.cat([x_t, h_prev], dim=-1)
        zr = torch.sigmoid(self.conv_gates(xh, supports))  # (B,N,2H)
        z, r = torch.split(zr, self.hidden_dim, dim=-1)

        xrh = torch.cat([x_t, r * h_prev], dim=-1)
        hc = torch.tanh(self.conv_cand(xrh, supports))     # (B,N,H)

        h = (1.0 - z) * h_prev + z * hc
        h = self.drop(h)
        return h


class DCRNNSeq2Seq(nn.Module):
    """
    DCRNN Seq2Seq baseline:
      Encoder: DCGRU over L steps with input_dim=F
      Decoder: DCGRU over H steps with input_dim=1 (y_prev), teacher forcing supported
      Output : (B,H,N)
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 64,
        horizon: int = 6,
        K: int = 2,
        num_supports: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.K = int(K)
        self.num_supports = int(num_supports)

        self.enc_cell = DCGRUCell(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            K=self.K,
            num_supports=self.num_supports,
            dropout=dropout,
        )

        self.dec_cell = DCGRUCell(
            num_nodes=self.num_nodes,
            input_dim=1,
            hidden_dim=self.hidden_dim,
            K=self.K,
            num_supports=self.num_supports,
            dropout=dropout,
        )

        self.proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, supports, teacher_forcing_y=None, teacher_forcing_ratio: float = 0.0):
        """
        x: (B,L,N,F)
        supports: list of (N,N) tensors
        teacher_forcing_y: (B,H,N) or None
        """
        if x.ndim != 4:
            raise RuntimeError(f"x must be (B,L,N,F), got {tuple(x.shape)}")
        B, L, N, Fdim = x.shape
        if N != self.num_nodes or Fdim != self.input_dim:
            raise RuntimeError(f"x shape mismatch: got N={N},F={Fdim}, expected N={self.num_nodes},F={self.input_dim}")

        # ---- Encoder ----
        h = torch.zeros((B, N, self.hidden_dim), device=x.device, dtype=x.dtype)
        for t in range(L):
            h = self.enc_cell(x[:, t], h, supports)

        # ---- Decoder ----
        y_hat = []
        y_prev = torch.zeros((B, N, 1), device=x.device, dtype=x.dtype)

        for t in range(self.horizon):
            h = self.dec_cell(y_prev, h, supports)
            y_t = self.proj(h).squeeze(-1)  # (B,N)
            y_hat.append(y_t)

            if (teacher_forcing_y is not None) and (teacher_forcing_ratio > 0.0):
                use_tf = (torch.rand(1, device=x.device).item() < float(teacher_forcing_ratio))
                if use_tf:
                    y_prev = teacher_forcing_y[:, t].unsqueeze(-1)
                else:
                    y_prev = y_t.unsqueeze(-1)
            else:
                y_prev = y_t.unsqueeze(-1)

        return torch.stack(y_hat, dim=1)  # (B,H,N)
