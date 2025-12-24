import torch
import torch.nn as nn


class STID(nn.Module):
    """
    STID-plus style model for your wind farm data.

    Input:  X (B, L, N, F)  - history window
    Output: Y_hat (B, H, N) - forecast horizon
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        d_model: int = 64,
        hidden_dim: int = 128,
        horizon: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        # -------- 1) 节点 identity embedding（空间身份） --------
        # 每台风机一个 embedding 向量
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)

        # -------- 2) 时间编码器：线性降维 + 2 层 GRU --------
        # X: (B,L,N,F) -> 先把 F 映射到较小维度 d_model，再送入 GRU
        self.in_proj = nn.Linear(input_dim, d_model)
        self.in_ln = nn.LayerNorm(d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # -------- 3) 时间注意力：对 L 个时间步做加权池化 --------
        # H_seq: (B*N, L, hidden_dim)
        # 先线性到 att_dim，再用一个向量做打分
        att_dim = hidden_dim
        self.att_lin = nn.Linear(hidden_dim, att_dim)
        self.att_vec = nn.Linear(att_dim, 1, bias=False)

        # -------- 4) 解码器：两层 MLP 输出 horizon 步 --------
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, horizon),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, L, N, F)
        return: (B, H, N)
        """
        B, L, N, F = X.shape
        assert N == self.num_nodes, f"num_nodes mismatch: got N={N}"

        # -------- step1: reshape 到 (B*N, L, F)，做输入线性映射 --------
        # X_bnlf: (B, N, L, F)
        X_bnlf = X.permute(0, 2, 1, 3).contiguous()
        X_seq = X_bnlf.view(B * N, L, F)  # (B*N, L, F)

        # 线性降维 + LayerNorm
        z = self.in_proj(X_seq)          # (B*N, L, d_model)
        z = self.in_ln(z)

        # -------- step2: 2 层 GRU 做时间编码 --------
        # H_seq: (B*N, L, hidden_dim)
        H_seq, _ = self.gru(z)

        # -------- step3: 时间注意力池化 --------
        # e: (B*N, L, att_dim) -> score: (B*N, L, 1)
        e = torch.tanh(self.att_lin(H_seq))
        score = self.att_vec(e).squeeze(-1)  # (B*N, L)
        alpha = torch.softmax(score, dim=1)  # 每个时间步的权重

        # context: (B*N, hidden_dim)
        context = torch.sum(H_seq * alpha.unsqueeze(-1), dim=1)
        # reshape 回 (B, N, hidden_dim)
        context = context.view(B, N, self.hidden_dim)

        # -------- step4: 节点 identity 融合 --------
        node_ids = torch.arange(N, device=X.device)  # (N,)
        node_e = self.node_emb(node_ids)             # (N, hidden_dim)
        node_e = node_e.unsqueeze(0).expand(B, -1, -1)  # (B, N, hidden_dim)

        # 简单相加（也可以 concat 再线性，这里先保持稳定）
        h = context + node_e  # (B, N, hidden_dim)

        # -------- step5: MLP 解码为 horizon 步 --------
        y_hat = self.mlp(h)  # (B, N, H)

        # 和 AGCRN 一样返回 (B, H, N)
        return y_hat.permute(0, 2, 1).contiguous()