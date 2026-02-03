import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adaptive_adj(E1, E2, topk=None):
    # E1,E2: (N, d)
    A = F.relu(E1 @ E2.t())  # (N,N)
    A = F.softmax(A, dim=-1)
    if topk is not None and topk < A.size(0):
        vals, idx = torch.topk(A, k=topk, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(-1, idx, 1.0)
        A = A * mask
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
    return A


class DiffusionGraphConv(nn.Module):
    """
    Diffusion GCN with supports [A, A^T], K steps.
    X: (B,N,Fin) -> (B,N,Fout)
    """
    def __init__(self, fin, fout, K=2):
        super().__init__()
        self.K = K
        self.lin = nn.Linear((2 * K + 1) * fin, fout)

    def forward(self, X, A):
        # A: (N,N)
        B, N, Fin = X.shape
        supports = [A, A.t()]
        out = [X]  # k=0
        for S in supports:
            Xk = X
            for _ in range(self.K):
                Xk = torch.einsum("nm,bmf->bnf", S, Xk)
                out.append(Xk)
        Z = torch.cat(out, dim=-1)  # (B,N,(2K+1)*Fin)
        return self.lin(Z)


class AGCRNCell(nn.Module):
    def __init__(self, fin, hidden, K=2, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.gc_z = DiffusionGraphConv(fin + hidden, hidden, K=K)
        self.gc_r = DiffusionGraphConv(fin + hidden, hidden, K=K)
        self.gc_h = DiffusionGraphConv(fin + hidden, hidden, K=K)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x_t, h_prev, A):
        # x_t: (B,N,fin), h_prev: (B,N,hidden)
        inp = torch.cat([x_t, h_prev], dim=-1)
        z = torch.sigmoid(self.gc_z(inp, A))
        r = torch.sigmoid(self.gc_r(inp, A))
        inp_r = torch.cat([x_t, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.gc_h(inp_r, A))
        h = (1 - z) * h_prev + z * h_tilde
        h = self.ln(h)
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class AGCRNSeq2SeqBaseline(nn.Module):
    """
    Encoder: uses full SCADA features (F=7)
    Decoder: autoregressive power only (Fin_dec=1)
    Output: (B,H,N)
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 10,
        horizon: int = 6,
        K: int = 2,
        topk: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.N = num_nodes
        self.F = input_dim
        self.H = horizon

        # adaptive graph embeddings
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.topk = topk

        self.enc_cell = AGCRNCell(fin=input_dim, hidden=hidden_dim, K=K, dropout=dropout)
        self.dec_cell = AGCRNCell(fin=1,         hidden=hidden_dim, K=K, dropout=dropout)

        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, X, teacher_forcing_y=None, teacher_forcing_ratio: float = 0.0):
        """
        X: (B,L,N,F)
        teacher_forcing_y: (B,H,N) optional
        """
        B, L, N, F = X.shape
        assert N == self.N and F == self.F

        A = build_adaptive_adj(self.E1, self.E2, topk=self.topk)

        # -------- Encoder --------
        h = torch.zeros(B, N, self.enc_cell.hidden, device=X.device)
        for t in range(L):
            h = self.enc_cell(X[:, t], h, A)

        # -------- Decoder --------
        # initial input = anchor power from last X step feature0
        y_prev = X[:, -1, :, 0].unsqueeze(-1)  # (B,N,1)

        preds = []
        for t in range(self.H):
            h = self.dec_cell(y_prev, h, A)
            y_t = self.out_proj(h).squeeze(-1)  # (B,N)
            preds.append(y_t)

            if (teacher_forcing_y is not None) and (teacher_forcing_ratio > 0.0):
                use_tf = (torch.rand(B, device=X.device) < teacher_forcing_ratio).float().unsqueeze(-1)
                y_in = use_tf * teacher_forcing_y[:, t, :] + (1.0 - use_tf) * y_t
            else:
                y_in = y_t
            y_prev = y_in.unsqueeze(-1)  # (B,N,1)

        Y_hat = torch.stack(preds, dim=1)  # (B,H,N)
        return Y_hat
