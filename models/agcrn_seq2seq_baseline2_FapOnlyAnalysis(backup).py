# models/agcrn_seq2seq_baseline2_FapOnlyAnalysis.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphConv(nn.Module):
    """
    Simple adaptive graph conv with K supports: [I, A, A^2, ...]
    x: (B,N,Cin) -> (B,N,Cout)
    """
    def __init__(self, cin: int, cout: int, K: int):
        super().__init__()
        self.cin = int(cin)
        self.cout = int(cout)
        self.K = int(K)
        self.lin = nn.Linear(self.cin * self.K, self.cout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B,N,C)
        B, N, C = x.shape
        outs = []
        xk = x
        I = torch.eye(N, device=x.device, dtype=x.dtype)
        Ak = I
        for k in range(self.K):
            if k == 0:
                Ak = I
            elif k == 1:
                Ak = A
            else:
                Ak = torch.matmul(Ak, A)
            outs.append(torch.einsum("nm,bmc->bnc", Ak, x))
        x_cat = torch.cat(outs, dim=-1)  # (B,N,C*K)
        return self.lin(x_cat)


class AGCRNCell(nn.Module):
    """
    GRU-like recurrent cell with adaptive graph conv.
    """
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, K: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.embed_dim = int(embed_dim)

        self.node_emb = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim) * 0.02)

        self.gc_z = AdaptiveGraphConv(self.input_dim + self.hidden_dim, self.hidden_dim, K=self.K)
        self.gc_r = AdaptiveGraphConv(self.input_dim + self.hidden_dim, self.hidden_dim, K=self.K)
        self.gc_h = AdaptiveGraphConv(self.input_dim + self.hidden_dim, self.hidden_dim, K=self.K)

        self.drop = nn.Dropout(float(dropout))

    def _adaptive_adj(self) -> torch.Tensor:
        # A = softmax(relu(E E^T))
        E = self.node_emb
        A = torch.matmul(E, E.t())
        A = F.relu(A)
        A = F.softmax(A, dim=1)
        return A

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B,N,input_dim)
        h_prev: (B,N,hidden_dim)
        """
        A = self._adaptive_adj()

        xh = torch.cat([x_t, h_prev], dim=-1)
        z = torch.sigmoid(self.gc_z(xh, A))
        r = torch.sigmoid(self.gc_r(xh, A))

        xrh = torch.cat([x_t, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.gc_h(xrh, A))

        h = (1.0 - z) * h_prev + z * h_tilde
        h = self.drop(h)
        return h


class AGCRNSeq2SeqBaseline(nn.Module):
    """
    Encoder:
      x: (B,L,N,input_dim) -> h_last: (B,N,hidden_dim)

    Decoder:
      step t input = [y_prev, exog_t] where
        y_prev: (B,N,1)
        exog_t: (B,N,exog_dim)  (optional)
      -> output y_t: (B,N)

    Returns:
      y_hat: (B,H,N)
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 10,
        horizon: int = 6,
        K: int = 2,
        topk=None,
        dropout: float = 0.0,
        exog_dim: int = 0,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.horizon = int(horizon)
        self.K = int(K)
        self.exog_dim = int(exog_dim)

        self.enc_cell = AGCRNCell(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            K=self.K,
            embed_dim=self.embed_dim,
            dropout=dropout,
        )

        dec_in = 1 + self.exog_dim
        self.dec_cell = AGCRNCell(
            num_nodes=self.num_nodes,
            input_dim=dec_in,
            hidden_dim=self.hidden_dim,
            K=self.K,
            embed_dim=self.embed_dim,
            dropout=dropout,
        )

        self.proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, exog=None, teacher_forcing_y=None, teacher_forcing_ratio: float = 0.0):
        """
        x: (B,L,N,input_dim)
        exog: (B,H,N,exog_dim) or None
        teacher_forcing_y: (B,H,N) or None
        """
        if x.ndim != 4:
            raise RuntimeError(f"x must be (B,L,N,D), got {tuple(x.shape)}")
        B, L, N, D = x.shape
        if N != self.num_nodes or D != self.input_dim:
            raise RuntimeError(f"x shape mismatch: got N={N},D={D}, expected N={self.num_nodes},D={self.input_dim}")

        if self.exog_dim > 0:
            if exog is None:
                raise RuntimeError("exog_dim>0 but exog is None")
            if exog.ndim != 4:
                raise RuntimeError(f"exog must be (B,H,N,E), got {tuple(exog.shape)}")
            if exog.shape[0] != B or exog.shape[1] != self.horizon or exog.shape[2] != N or exog.shape[3] != self.exog_dim:
                raise RuntimeError(f"exog shape mismatch, got {tuple(exog.shape)} expected (B,{self.horizon},N,{self.exog_dim})")

        h = torch.zeros((B, N, self.hidden_dim), device=x.device, dtype=x.dtype)
        for t in range(L):
            h = self.enc_cell(x[:, t], h)

        y_hat = []
        y_prev = torch.zeros((B, N, 1), device=x.device, dtype=x.dtype)

        for t in range(self.horizon):
            if self.exog_dim > 0:
                ex_t = exog[:, t]  # (B,N,E)
                dec_in = torch.cat([y_prev, ex_t], dim=-1)
            else:
                dec_in = y_prev

            h = self.dec_cell(dec_in, h)
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

        y_hat = torch.stack(y_hat, dim=1)  # (B,H,N)
        return y_hat
