# models/agcrn_bt.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    """Adaptive Vertex-wise Graph Convolution (from BasicTS v0.5.8)."""

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.cheb_k = cheb_k
        # [embed_dim, cheb_k, dim_in, dim_out]
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)
        )
        # [embed_dim, dim_out]
        self.bias_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, dim_out)
        )

    def forward(self, x, node_embeddings):
        """
        x: [B, N, C_in]
        node_embeddings: [N, D]
        return: [B, N, C_out]
        """
        B, N, _ = x.shape
        device = x.device

        # 1) 自适应邻接矩阵 A
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.T)), dim=1
        )                           # [N, N]

        # 2) Chebyshev 多项式基
        support_set = [
            torch.eye(N, device=device),
            supports
        ]
        for k in range(2, self.cheb_k):
            support_set.append(
                2 * supports @ support_set[-1] - support_set[-2]
            )
        supports_stack = torch.stack(support_set, dim=0)  # [K, N, N]

        # 3) 根据 embedding 生成每个节点自己的卷积核 / bias
        # weights: [N, K, dim_in, dim_out]
        weights = torch.einsum(
            'nd,dkio->nkio', node_embeddings, self.weights_pool
        )
        # bias: [N, dim_out]
        bias = node_embeddings @ self.bias_pool

        # 4) 图卷积：先扩展 Cheb 基，再乘以权重
        # x_g: [B, K, N, dim_in]
        x_g = torch.einsum("knm,bmc->bknc", supports_stack, x)
        # -> [B, N, K, dim_in]
        x_g = x_g.permute(0, 2, 1, 3)
        # -> [B, N, dim_out]
        x_gconv = torch.einsum("bnki,nkio->bno", x_g, weights) + bias

        return x_gconv


class AGCRNCell(nn.Module):
    """Graph-GRU cell (from BasicTS v0.5.8)."""

    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out

        self.gate = AVWGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: [B, N, C_in]
        # state: [B, N, H]
        state = state.to(x.device)

        input_and_state = torch.cat([x, state], dim=-1)          # [B, N, C_in+H]
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)

        candidate = torch.cat([x, z * state], dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))

        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size, device):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device)


class AVWDCRNN(nn.Module):
    """多层 AGCRN 编码器 (DCRNN 风格)."""

    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super().__init__()
        assert num_layers >= 1
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers

        cells = [AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim)]
        for _ in range(1, num_layers):
            cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
        self.dcrnn_cells = nn.ModuleList(cells)

    def forward(self, x, init_state, node_embeddings):
        """
        x: [B, T, N, C_in]
        init_state: [num_layers, B, N, hidden_dim]
        """
        assert x.shape[2] == self.node_num
        seq_len = x.shape[1]
        current_inputs = x

        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_len):
                state = self.dcrnn_cells[i](current_inputs[:, t], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)  # [B, T, N, H]

        return current_inputs, torch.stack(output_hidden, dim=0)

    def init_hidden(self, batch_size, device):
        states = [
            cell.init_hidden_state(batch_size, device) for cell in self.dcrnn_cells
        ]
        return torch.stack(states, dim=0)  # [L, B, N, H]


class AGCRN(nn.Module):
    """
    AGCRN with simplified forward(history) interface for your project.
    输入:  x: [B, L, N, F]
    输出:  y_hat: [B, H, N]  (这里 output_dim=1, squeeze 掉最后一维)
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        rnn_units: int,
        horizon: int,
        num_layers: int = 1,
        embed_dim: int = 8,
        cheb_k: int = 3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = rnn_units
        self.horizon = horizon
        self.output_dim = 1  # 预测功率一个标量

        # 可学习节点 embedding
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embed_dim)
        )

        self.encoder = AVWDCRNN(
            node_num=num_nodes,
            dim_in=input_dim,
            dim_out=rnn_units,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

        # 1x1 卷积预测头
        self.end_conv = nn.Conv2d(
            in_channels=1,
            out_channels=horizon * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        )

        self.init_param()

    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, x):
        """
        x: [B, L, N, F]
        return: [B, H, N]
        """
        B = x.size(0)
        device = x.device
        init_state = self.encoder.init_hidden(B, device)  # [L, B, N, H]

        enc_out, _ = self.encoder(x, init_state, self.node_embeddings)  # [B, L, N, H]
        last_hidden = enc_out[:, -1:, :, :]  # [B, 1, N, H]

        out = self.end_conv(last_hidden)  # [B, H*1, N, 1]
        out = out.squeeze(-1)             # [B, H*1, N]
        out = out.view(B, self.horizon, self.num_nodes)  # [B, H, N]
        return out