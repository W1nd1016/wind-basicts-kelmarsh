# models/agcrn_seq2seq_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AGCN(nn.Module):
    """
    Adaptive graph convolution using learned node embeddings.
    """
    def __init__(self, num_nodes: int, in_dim: int, out_dim: int, embed_dim: int, K: int = 2):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.embed_dim = int(embed_dim)
        self.K = int(K)

        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(embed_dim, num_nodes) * 0.1)

        # weights for K hops (including 0-hop)
        self.W = nn.Parameter(torch.randn(K + 1, in_dim, out_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def _adaptive_adj(self):
        # A = softmax(ReLU(E1 E2))
        A = F.softmax(F.relu(self.node_emb1 @ self.node_emb2), dim=1)  # (N,N)
        return A

    def forward(self, x):
        """
        x: (B,N,in_dim)
        return: (B,N,out_dim)
        """
        A = self._adaptive_adj()  # (N,N)
        supports = [torch.eye(self.num_nodes, device=x.device, dtype=x.dtype), A]
        # build higher-order supports A^k
        for k in range(2, self.K + 1):
            supports.append(supports[-1] @ A)

        out = 0.0
        for k, S in enumerate(supports):
            # (B,N,in_dim) -> (B,N,in_dim) via graph mix
            xk = torch.einsum("nm,bmi->bni", S, x)
            out = out + torch.einsum("bni,io->bno", xk, self.W[k])
        out = out + self.b
        return out

class AGCRNCell(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, embed_dim, K=2):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.gc_z = AGCN(num_nodes, input_dim + hidden_dim, hidden_dim, embed_dim, K=K)
        self.gc_r = AGCN(num_nodes, input_dim + hidden_dim, hidden_dim, embed_dim, K=K)
        self.gc_h = AGCN(num_nodes, input_dim + hidden_dim, hidden_dim, embed_dim, K=K)

    def forward(self, x, h):
        """
        x: (B,N,input_dim)
        h: (B,N,hidden_dim)
        """
        xh = torch.cat([x, h], dim=-1)
        z = torch.sigmoid(self.gc_z(xh))
        r = torch.sigmoid(self.gc_r(xh))

        xrh = torch.cat([x, r * h], dim=-1)
        hc = torch.tanh(self.gc_h(xrh))

        h_new = (1.0 - z) * h + z * hc
        return h_new

class AGCRNSeq2SeqBaseline(nn.Module):
    """
    Encoder-decoder AGCRN.
    Input:  x (B,L,N,input_dim)
    Output: yhat (B,H,N)
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 10,
        horizon: int = 6,
        K: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.dropout = float(dropout)

        self.encoder_cell = AGCRNCell(num_nodes, input_dim, hidden_dim, embed_dim, K=K)
        self.decoder_cell = AGCRNCell(num_nodes, 1, hidden_dim, embed_dim, K=K)  # decoder consumes previous y (1 dim)

        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, teacher_forcing_y=None, teacher_forcing_ratio: float = 0.0):
        """
        x: (B,L,N,input_dim)
        teacher_forcing_y: (B,H,N) normalized, optional
        """
        B, L, N, D = x.shape
        assert N == self.num_nodes

        # ----- encoder -----
        h = torch.zeros((B, N, self.hidden_dim), device=x.device, dtype=x.dtype)
        for t in range(L):
            xt = x[:, t, :, :]
            h = self.encoder_cell(xt, h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # ----- decoder -----
        yhat = []
        # start token: 0
        y_prev = torch.zeros((B, N, 1), device=x.device, dtype=x.dtype)

        for i in range(self.horizon):
            h = self.decoder_cell(y_prev, h)
            out = self.proj(h).squeeze(-1)  # (B,N)
            yhat.append(out)

            if teacher_forcing_y is not None and self.training and teacher_forcing_ratio > 0:
                use_tf = (torch.rand((B, 1, 1), device=x.device) < teacher_forcing_ratio).float()
                gt = teacher_forcing_y[:, i, :].unsqueeze(-1)  # (B,N,1)
                y_prev = use_tf * gt + (1.0 - use_tf) * out.unsqueeze(-1)
            else:
                y_prev = out.unsqueeze(-1)

        return torch.stack(yhat, dim=1)  # (B,H,N)
