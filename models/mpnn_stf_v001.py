# models/mpnn_stf_v001.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNNLayer(nn.Module):
    """
    Standard message passing layer:
      message_ij = MLP([h_i, h_j])
      agg_i = sum_j A_ij * message_ij
      h_i <- GRUCell(agg_i, h_i)

    h: (B, N, D)
    A: (N, N) row-normalized adjacency (float)
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.upd = nn.GRUCell(input_size=dim, hidden_size=dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        assert A.shape == (N, N), f"A shape {A.shape} != (N,N)=({N},{N})"

        # Build all pair (i,j) in a vectorized way:
        # hi: (B,N,1,D), hj: (B,1,N,D) -> pair: (B,N,N,2D)
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        pair = torch.cat([hi, hj], dim=-1)  # (B,N,N,2D)

        msg = self.msg_mlp(pair)  # (B,N,N,D)

        # weighted aggregation by adjacency
        Aw = A.view(1, N, N, 1)  # (1,N,N,1)
        agg = (msg * Aw).sum(dim=2)  # sum over j -> (B,N,D)

        # GRU update node-wise
        h0 = h.reshape(B * N, D)
        agg0 = agg.reshape(B * N, D)
        h_new = self.upd(agg0, h0).reshape(B, N, D)

        h_new = self.ln(h_new)
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)
        return h_new


class MPNN_STF_v001(nn.Module):
    """
    Paper-grade baseline for SCADA-only STF:
      - Temporal encoder per node: GRU over L steps
      - Spatial encoder: K-step MPNN over turbine graph (adj.npy)
      - Output: horizon H per node

    Input : x (B, L, N, F)
    Output: y_hat (B, H, N)
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        horizon: int,
        hidden_dim: int = 64,
        gru_layers: int = 1,
        mp_steps: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.mp_steps = mp_steps

        # temporal encoder (shared across nodes)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.in_ln = nn.LayerNorm(hidden_dim)

        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # message passing layers
        self.mp = nn.ModuleList([MPNNLayer(hidden_dim, dropout=dropout) for _ in range(mp_steps)])

        # horizon head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: (B,L,N,F)
        A: (N,N)
        return: (B,H,N)
        """
        B, L, N, F = x.shape
        assert N == self.num_nodes, f"N={N} != num_nodes={self.num_nodes}"
        assert F == self.input_dim, f"F={F} != input_dim={self.input_dim}"

        # (B,L,N,F) -> (B,N,L,F) -> (B*N,L,F)
        x_bnlf = x.permute(0, 2, 1, 3).contiguous()
        x_seq = x_bnlf.view(B * N, L, F)

        z = self.in_proj(x_seq)     # (B*N,L,D)
        z = self.in_ln(z)

        out, _ = self.temporal_gru(z)      # (B*N,L,D)
        h = out[:, -1, :].view(B, N, self.hidden_dim)  # (B,N,D)

        # K-step message passing
        for layer in self.mp:
            h = layer(h, A)

        # decode to horizon
        y_hat = self.head(h)  # (B,N,H)
        return y_hat.permute(0, 2, 1).contiguous()  # (B,H,N)
