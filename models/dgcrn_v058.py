# models/dgcrn_v058.py
# 通用 DGCRN 实现，支持 encoder 用 55 维特征，decoder 只用 2 维（预测值 + 时间）
# 你不需要再在模型里裁剪特征，裁剪逻辑全部在训练脚本外部控制（现在已经是直接用 55 维）

import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ======================= 基本图卷积模块 =======================

class gconv_RNN(nn.Module):
    def __init__(self):
        super(gconv_RNN, self).__init__()

    def forward(self, x, A):
        # x: [B, N, C], A: [B, N, N]
        x = torch.einsum("bnc,bnm->bmc", (x, A))
        return x.contiguous()


class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()

    def forward(self, x, A):
        # x: [B, N, C], A: [N, N]
        A = A.to(x.device)
        x = torch.einsum("bnc,nm->bmc", (x, A))
        return x.contiguous()


class gcn(nn.Module):
    """
    dims:
      - type == 'RNN'   : dims = [in_dim, out_dim]
      - type == 'hyper' : dims = [in_dim, h1, h2, h3]
    """

    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None):
        super(gcn, self).__init__()
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

        if type == "RNN":
            self.gconv = gconv_RNN()
            self.gconv_preA = gconv_hyper()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])

        elif type == "hyper":
            self.gconv = gconv_hyper()
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("fc1", nn.Linear((gdep + 1) * dims[0], dims[1])),
                        ("sigmoid1", nn.Sigmoid()),
                        ("fc2", nn.Linear(dims[1], dims[2])),
                        ("sigmoid2", nn.Sigmoid()),
                        ("fc3", nn.Linear(dims[2], dims[3])),
                    ]
                )
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: [B, N, Cin]
        如果 type == 'RNN'，adj 是 [adj_dynamic, adj_pre] 列表
        如果 type == 'hyper'，adj 是单个 [N, N] 矩阵
        """
        h = x
        out = [h]

        if self.type_GNN == "RNN":
            for _ in range(self.gdep):
                h = (
                    self.alpha * x
                    + self.beta * self.gconv(h, adj[0])
                    + self.gamma * self.gconv_preA(h, adj[1])
                )
                out.append(h)
        else:
            for _ in range(self.gdep):
                h = self.alpha * x + self.gamma * self.gconv(h, adj)
                out.append(h)

        ho = torch.cat(out, dim=-1)  # [..., (gdep+1)*Cin]
        ho = self.mlp(ho)
        ho = self.dropout(ho)
        return ho


# ======================= DGCRN 主体 =======================

class DGCRN(nn.Module):
    """
    Paper: Dynamic Graph Convolutional Recurrent Network for Traffic Prediction
    Task: Spatial-Temporal Forecasting
    """

    def __init__(
        self,
        gcn_depth,
        num_nodes,
        predefined_A=None,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        middle_dim=2,
        seq_length=12,
        in_dim=2,
        list_weight=[0.05, 0.95, 0.95],
        tanhalpha=3,
        cl_decay_steps=4000,
        rnn_size=64,
        hyperGNN_dim=16,
    ):
        super(DGCRN, self).__init__()

        self.output_dim = 1
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.seq_length = seq_length

        self.rnn_size = rnn_size
        self.hidden_size = self.rnn_size
        self.in_dim = in_dim  # 这里等于 55（历史输入特征维度）

        # ---------- 节点 embedding ----------
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)

        self.idx = torch.arange(self.num_nodes)

        # ---------- 超图 GCN 维度 ----------
        # encoder: x 是 [B, N, 55]，hyper_input = [x, h] => 55 + 64 = 119
        dims_hyper_enc = [
            self.hidden_size + self.in_dim,  # 64 + 55 = 119
            hyperGNN_dim,
            middle_dim,
            node_dim,
        ]
        # decoder: x 是 [B, N, 2]（预测值 + 时间），hyper_input = [x, h] => 2 + 64 = 66
        dims_hyper_dec = [
            self.hidden_size + 2,  # 64 + 2 = 66
            hyperGNN_dim,
            middle_dim,
            node_dim,
        ]

        # ---------- RNN 内部 GCN 维度 ----------
        # encoder: combined = [x(55), h(64)] => 119
        dims_rnn_enc = [self.in_dim + self.hidden_size, self.hidden_size]
        # decoder: combined = [x(2), h(64)] => 66
        dims_rnn_dec = [2 + self.hidden_size, self.hidden_size]

        # ---------- hyper GCN（encoder） ----------
        self.GCN1_tg = gcn(
            dims_hyper_enc, gcn_depth, dropout, *list_weight, type="hyper"
        )
        self.GCN2_tg = gcn(
            dims_hyper_enc, gcn_depth, dropout, *list_weight, type="hyper"
        )
        self.GCN1_tg_1 = gcn(
            dims_hyper_enc, gcn_depth, dropout, *list_weight, type="hyper"
        )
        self.GCN2_tg_1 = gcn(
            dims_hyper_enc, gcn_depth, dropout, *list_weight, type="hyper"
        )

        # ---------- hyper GCN（decoder） ----------
        self.GCN1_tg_de = gcn(
            dims_hyper_dec, gcn_depth, dropout, *list_weight, type="hyper"
        )
        self.GCN2_tg_de = gcn(
            dims_hyper_dec, gcn_depth, dropout, *list_weight, type="hyper"
        )
        self.GCN1_tg_de_1 = gcn(
            dims_hyper_dec, gcn_depth, dropout, *list_weight, type="hyper"
        )
        self.GCN2_tg_de_1 = gcn(
            dims_hyper_dec, gcn_depth, dropout, *list_weight, type="hyper"
        )

        # ---------- RNN GCN（encoder） ----------
        self.gz1 = gcn(dims_rnn_enc, gcn_depth, dropout, *list_weight, type="RNN")
        self.gz2 = gcn(dims_rnn_enc, gcn_depth, dropout, *list_weight, type="RNN")
        self.gr1 = gcn(dims_rnn_enc, gcn_depth, dropout, *list_weight, type="RNN")
        self.gr2 = gcn(dims_rnn_enc, gcn_depth, dropout, *list_weight, type="RNN")
        self.gc1 = gcn(dims_rnn_enc, gcn_depth, dropout, *list_weight, type="RNN")
        self.gc2 = gcn(dims_rnn_enc, gcn_depth, dropout, *list_weight, type="RNN")

        # ---------- RNN GCN（decoder） ----------
        self.gz1_de = gcn(dims_rnn_dec, gcn_depth, dropout, *list_weight, type="RNN")
        self.gz2_de = gcn(dims_rnn_dec, gcn_depth, dropout, *list_weight, type="RNN")
        self.gr1_de = gcn(dims_rnn_dec, gcn_depth, dropout, *list_weight, type="RNN")
        self.gr2_de = gcn(dims_rnn_dec, gcn_depth, dropout, *list_weight, type="RNN")
        self.gc1_de = gcn(dims_rnn_dec, gcn_depth, dropout, *list_weight, type="RNN")
        self.gc2_de = gcn(dims_rnn_dec, gcn_depth, dropout, *list_weight, type="RNN")

        # 输出层
        self.fc_final = nn.Linear(self.hidden_size, self.output_dim)

        self.alpha = tanhalpha
        self.k = subgraph_size

        # curriculum learning
        self.use_curriculum_learning = True
        self.cl_decay_steps = cl_decay_steps
        self.gcn_depth = gcn_depth

    # ======================= 工具函数 =======================

    def preprocessing(self, adj, predefined_A):
        # adj: [B, N, N] 动态图
        adj = adj + torch.eye(self.num_nodes).to(adj.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_A]

    def initHidden(self, batch_size, hidden_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            nn.init.orthogonal_(Hidden_State)
            nn.init.orthogonal_(Cell_State)
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
        return Hidden_State, Cell_State

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)
        )

    # ======================= 单步 RNN（encoder / decoder 共用） =======================

    def step(self, input, Hidden_State, Cell_State, predefined_A, type="encoder", i=None):
        """
        input: [B, N, Cx]
        encoder: Cx = 55
        decoder: Cx = 2
        Hidden_State/Cell_State: [B*N, H]
        predefined_A: [A1, A2], each [N, N]
        """
        x = input  # [B, N, Cx]
        B, N, Cx = x.shape

        nodevec1 = self.emb1(self.idx)  # [N, node_dim]
        nodevec2 = self.emb2(self.idx)  # [N, node_dim]

        h_reshaped = Hidden_State.view(-1, self.num_nodes, self.hidden_size)  # [B, N, H]
        hyper_input = torch.cat((x, h_reshaped), dim=2)  # [B, N, Cx+H]

        if type == "encoder":
            filter1 = self.GCN1_tg(hyper_input, predefined_A[0]) + self.GCN1_tg_1(
                hyper_input, predefined_A[1]
            )
            filter2 = self.GCN2_tg(hyper_input, predefined_A[0]) + self.GCN2_tg_1(
                hyper_input, predefined_A[1]
            )
        else:
            filter1 = self.GCN1_tg_de(hyper_input, predefined_A[0]) + self.GCN1_tg_de_1(
                hyper_input, predefined_A[1]
            )
            filter2 = self.GCN2_tg_de(hyper_input, predefined_A[0]) + self.GCN2_tg_de_1(
                hyper_input, predefined_A[1]
            )

        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))  # [B, N, node_dim]
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))

        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - torch.matmul(
            nodevec2, nodevec1.transpose(2, 1)
        )  # [B, N, N]
        adj = F.relu(torch.tanh(self.alpha * a))  # [B, N, N]

        adp = self.preprocessing(adj, predefined_A[0])
        adpT = self.preprocessing(adj.transpose(1, 2), predefined_A[1])

        Hidden_State = Hidden_State.view(-1, self.num_nodes, self.hidden_size)
        Cell_State = Cell_State.view(-1, self.num_nodes, self.hidden_size)

        combined = torch.cat((x, Hidden_State), dim=-1)  # [B, N, Cx+H]

        if type == "encoder":
            z = torch.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
            r = torch.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))
            temp = torch.cat((x, torch.mul(r, Hidden_State)), dim=-1)
            Cell_State = torch.tanh(self.gc1(temp, adp) + self.gc2(temp, adpT))
        else:
            z = torch.sigmoid(self.gz1_de(combined, adp) + self.gz2_de(combined, adpT))
            r = torch.sigmoid(self.gr1_de(combined, adp) + self.gr2_de(combined, adpT))
            temp = torch.cat((x, torch.mul(r, Hidden_State)), dim=-1)
            Cell_State = torch.tanh(self.gc1_de(temp, adp) + self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State) + torch.mul(1 - z, Cell_State)

        return Hidden_State.view(-1, self.hidden_size), Cell_State.view(-1, self.hidden_size)

    # ======================= 前向传播 =======================

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        """
        history_data: [B, L, N, C]   这里 C = in_dim = 55
        future_data:  [B, H, N, 2]   通道 0 = 真实 P，通道 1 = 时间
        kwargs:
            task_level: int, 等于 H
        返回:
            outputs_final: [B, L, N, 1]
        """
        task_level = kwargs["task_level"]

        # [B, L, N, C] -> [B, C, N, L]
        input = history_data.transpose(1, 3).contiguous()
        ycl = future_data.transpose(1, 3).contiguous()  # [B, 2, N, H]

        self.idx = self.idx.to(input.device)
        predefined_A = self.predefined_A

        x = input
        B = x.size(0)

        Hidden_State, Cell_State = self.initHidden(B * self.num_nodes, self.hidden_size)
        Hidden_State = Hidden_State.to(input.device)
        Cell_State = Cell_State.to(input.device)

        outputs = None

        # ---------- Encoder ----------
        for i in range(self.seq_length):
            # x[..., i]: [B, C, N] -> [B, N, C]
            x_t = x[..., i].transpose(1, 2).contiguous()
            Hidden_State, Cell_State = self.step(
                x_t, Hidden_State, Cell_State, predefined_A, type="encoder", i=i
            )

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)  # [B, 1, H]
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), dim=1)  # [B, L, H]

        # ---------- Decoder ----------
        go_symbol = torch.zeros((B, self.output_dim, self.num_nodes), device=input.device)  # [B, 1, N]
        timeofday = ycl[:, [1], :, :]  # [B, 1, N, H]

        decoder_input = go_symbol  # [B, 1, N]
        outputs_final = []

        for i in range(task_level):
            # 拼接当前预测值和时间特征 -> [B, 2, N]
            decoder_input_cat = torch.cat(
                [decoder_input, timeofday[..., i]], dim=1
            )  # [B, 2, N]
            decoder_input_t = decoder_input_cat.transpose(1, 2).contiguous()  # [B, N, 2]

            Hidden_State, Cell_State = self.step(
                decoder_input_t, Hidden_State, Cell_State, predefined_A, type="decoder", i=None
            )

            decoder_output = self.fc_final(Hidden_State)  # [B*N, 1]
            decoder_input = (
                decoder_output.view(B, self.num_nodes, self.output_dim).transpose(1, 2)
            )  # [B, 1, N]
            outputs_final.append(decoder_output)

            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batch_seen):
                    decoder_input = ycl[:, :1, :, i]  # teacher forcing：用真实值

        outputs_final = torch.stack(outputs_final, dim=1)  # [B*N, H, 1]
        outputs_final = (
            outputs_final.view(B, self.num_nodes, task_level, self.output_dim)
            .transpose(1, 2)
            .contiguous()
        )  # [B, H, N, 1]

        # 后面 L - H 步补 0
        random_predict = torch.zeros(
            B,
            self.seq_length - task_level,
            self.num_nodes,
            self.output_dim,
            device=outputs_final.device,
        )
        outputs_final = torch.cat([outputs_final, random_predict], dim=1)  # [B, L, N, 1]

        return outputs_final