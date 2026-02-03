import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adaptive_adj(E1, E2, topk=None):
    """
    Build adaptive adjacency A from learnable node embeddings.
    E1,E2: (N, d)
    Return:
      A: (N, N) row-normalized
    """
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
        """
        X: (B,N,Fin)
        A: (N,N)
        """
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
    """
    One recurrent cell with diffusion graph convolution gates (GRU-like).
    """
    def __init__(self, fin, hidden, K=2, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.gc_z = DiffusionGraphConv(fin + hidden, hidden, K=K)
        self.gc_r = DiffusionGraphConv(fin + hidden, hidden, K=K)
        self.gc_h = DiffusionGraphConv(fin + hidden, hidden, K=K)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x_t, h_prev, A):
        """
        x_t:   (B,N,fin)
        h_prev:(B,N,hidden)
        A:     (N,N)
        """
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
    ===== Direct Multi-Horizon (NON-autoregressive) AGCRN =====

    你的原版是 Seq2Seq 自回归 decoder：
      - 每小时预测一次，并把上一小时预测喂回去 (iterative)

    这个版本改成“直接输出 H 步”：
      - Encoder 跑完 L 步历史，得到最后隐状态 h_last: (B,N,hidden)
      - 通过一个 head 一次性输出 H 个未来步： (B,H,N)
      - 不再使用 decoder、不需要 y_prev、不使用 teacher forcing

    输入:
      X: (B,L,N,F)

    输出:
      Y_hat: (B,H,N)
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
        head_hidden: int = 128,
    ):
        super().__init__()
        self.N = int(num_nodes)
        self.F = int(input_dim)
        self.H = int(horizon)

        # adaptive graph embeddings -> adjacency
        self.E1 = nn.Parameter(torch.randn(self.N, embed_dim))
        self.E2 = nn.Parameter(torch.randn(self.N, embed_dim))
        self.topk = topk

        # encoder over history
        self.enc_cell = AGCRNCell(fin=self.F, hidden=hidden_dim, K=K, dropout=dropout)

        # direct multi-horizon head: (B,N,hidden) -> (B,N,H)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.H),
        )

    def forward(self, X, teacher_forcing_y=None, teacher_forcing_ratio: float = 0.0):
        """
        X: (B,L,N,F)
        teacher_forcing_y / teacher_forcing_ratio:
          - 为了兼容你现有 wrapper/train_loop 的调用签名保留
          - 本版本不使用它们（因为没有自回归 decoder）
        """
        B, L, N, Fdim = X.shape
        if N != self.N or Fdim != self.F:
            raise RuntimeError(f"shape mismatch: got (N={N},F={Fdim}), expect (N={self.N},F={self.F})")

        A = build_adaptive_adj(self.E1, self.E2, topk=self.topk)

        # -------- Encoder over history --------
        h = torch.zeros(B, N, self.enc_cell.hidden, device=X.device)
        for t in range(L):
            h = self.enc_cell(X[:, t], h, A)

        # -------- Direct multi-horizon output --------
        # y: (B,N,H) -> (B,H,N)
        y = self.head(h)
        Y_hat = y.permute(0, 2, 1).contiguous()
        return Y_hat
