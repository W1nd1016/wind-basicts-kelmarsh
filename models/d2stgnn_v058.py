# models/d2stgnn_v058.py
# 适配自 BasicTS v0.5.8 的 D2STGNN，实现为你自己的 pipeline 版本
# 使用方式：和 AGCRN_v058 / DGCRN_v058 一样，在训练脚本中：
#   from models.d2stgnn_v058 import D2STGNN_v058
#   model = D2STGNN_v058(num_nodes=6, input_dim=55, seq_len=input_len, horizon=output_len, adjs=[adj_tensor])

import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== Decouple: Spatial gate =====================

class EstimationGate(nn.Module):
    """空间门控模块，用节点/时间 embedding 估计一个 [0,1] 的空间权重，对 X 做加权。"""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim, input_seq_len):
        super().__init__()
        self.FC1 = nn.Linear(2 * node_emb_dim + 2 * time_emb_dim, hidden_dim)
        self.act = nn.ReLU()
        self.FC2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_embedding1, node_embedding2, T_D, D_W, X):
        # T_D, D_W: [B, L, N, Dt]
        B, L, N, Dt = T_D.shape
        spa_feat = torch.cat(
            [
                T_D,
                D_W,
                node_embedding1.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
                node_embedding2.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            ],
            dim=-1,
        )  # [B, L, N, 2Dt+2Dnode]

        hidden = self.act(self.FC1(spa_feat))
        gate = torch.sigmoid(self.FC2(hidden))[:, -X.shape[1] :, :, :]  # 对齐时间长度
        return X * gate


# ===================== Residual decomposition =====================

class ResidualDecomp(nn.Module):
    """x - f(y) 然后做 LayerNorm 的残差分解。"""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u


# ===================== Diffusion block =====================
class STLocalizedConv(nn.Module):
    """
    时空局部图卷积：
    - 时间维用 kernel_size = k_t 的滑窗
    - 空间维对 (预定义图 / 静态自适应图) 做多阶扩散
    """

    def __init__(self, hidden_dim, pre_defined_graph=None, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args["k_s"]
        self.k_t = model_args["k_t"]
        self.hidden_dim = hidden_dim

        # 图开关
        self.use_predefined_graph = bool(use_pre)
        self.use_dynamic_hidden_graph = bool(dy_graph)      # 现在是 False
        self.use_static__hidden_graph = bool(sta_graph)     # True

        # 预定义图：list[Tensor(N, N)]
        if pre_defined_graph is None:
            self.pre_defined_graph = []
        else:
            self.pre_defined_graph = [
                torch.as_tensor(g, dtype=torch.float32) for g in pre_defined_graph
            ]

        self.dropout = nn.Dropout(model_args["dropout"])

        # 先对预定义图做多阶扩散 & 时空本地化 => list[Tensor(N, k_t * N)]
        if self.use_predefined_graph and len(self.pre_defined_graph) > 0:
            self.pre_defined_graph = self.get_graph(self.pre_defined_graph)
        else:
            self.pre_defined_graph = []

        # 时间维上的 1×1 线性层（相当于对每个滑窗做线性变换）
        self.fc_list_updt = nn.Linear(self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)

        # 图卷积后的线性层：**输入和输出都是 hidden_dim=64**
        # 我们把多个图的结果相加（sum），而不是在特征维拼接
        self.gcn_updt = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

    def get_graph(self, support: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        只对静态图（预定义 / 自适应）做多阶扩散，并展开成时空局部图：
        输入：support 中每个图是 [N, N]
        输出：list 中每个图是 [N, k_t * N]
        """
        assert len(support) > 0
        device = support[0].device
        N = support[0].shape[0]
        mask = 1 - torch.eye(N, device=device)

        graph_ordered = []
        for graph in support:
            k_1_order = graph * mask
            graph_ordered.append(k_1_order)
            for _ in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order) * mask
                graph_ordered.append(k_1_order)

        st_local_graph = []
        for graph in graph_ordered:
            # [N, N] -> [N, k_t, N] -> [N, k_t * N]
            graph = graph.unsqueeze(1).expand(N, self.k_t, N)
            graph = graph.reshape(N, self.k_t * N)
            st_local_graph.append(graph)
        return st_local_graph

    def gconv(self, support: list[torch.Tensor], X_k: torch.Tensor, X_0: torch.Tensor) -> torch.Tensor:
        """
        support : List[Tensor]，每个是 [N, k_t * N]
        X_k     : [B, T, k_t * N, D]
        X_0     : [B, T, N, D]
        返回    : [B, T, N, D]
        """
        B, T, N, D = X_0.shape
        device = X_0.device

        # 起始项：不卷积的 “自环” 特征
        out_sum = X_0.clone()

        for graph in support:
            # graph: [N, k_t * N]
            if graph.dim() != 2:
                # 以防万一，如果是 3D/4D，就 squeeze 成 2D
                g = graph.to(device)
                while g.dim() > 2:
                    g = g[0]
            else:
                g = graph.to(device)

            # g: [N, K]，X_k: [B, T, K, D]
            # => H_k: [B, T, N, D]
            H_k = torch.einsum("nk, btkd -> btnd", g, X_k)
            out_sum = out_sum + H_k

        # out_sum: [B, T, N, D]
        out = self.gcn_updt(out_sum)     # 线性映射 (D -> D)
        out = self.dropout(out)
        return out

    def forward(self, X: torch.Tensor, dynamic_graph, static_graph):
        """
        X: [B, L, N, D]
        dynamic_graph: list[...]（当前关闭）
        static_graph: list[Tensor(N, N)]，由节点 embedding 学出来的自适应图
        """
        # 时间局部：取长度为 k_t 的滑窗
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        B, seq_len, N, k_t, D = X.shape

        # 构造支持图列表
        support = []
        # 预定义图（来自 adj.npy）
        if self.use_predefined_graph and len(self.pre_defined_graph) > 0:
            support += self.pre_defined_graph
        # 动态图（当前关闭）
        if self.use_dynamic_hidden_graph and dynamic_graph is not None:
            support += dynamic_graph
        # 静态自适应图（E_u, E_d 学出来）
        if self.use_static__hidden_graph and static_graph is not None and len(static_graph) > 0:
            support += self.get_graph(static_graph)

        # 时间维 1×1 线性层
        X = X.reshape(B, seq_len, N, k_t * D)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(B, seq_len, N, k_t, D)

        X_0 = torch.mean(out, dim=-2)                   # [B, T, N, D]
        X_k = out.transpose(-3, -2).reshape(B, seq_len, k_t * N, D)  # [B, T, k_t*N, D]

        hidden = self.gconv(support, X_k, X_0)          # [B, T, N, D]
        return hidden


class DifForecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args["k_t"]
        self.output_seq_len = model_args["seq_length"]
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        self.model_args = model_args

    def forward(self, X, H, st_l_conv, dynamic_graph, static_graph):
        # X, H: [B, L, N, D]
        B, seq_len_remain, N, D = H.shape
        B, seq_len_input, N, D = X.shape

        predict = []
        history = X
        predict.append(H[:, -1, :, :].unsqueeze(1))  # seed

        for _ in range(int(self.output_seq_len / self.model_args["gap"]) - 1):
            _1 = predict[-self.k_t :]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            predict.append(st_l_conv(_1, dynamic_graph, static_graph))

        predict = torch.cat(predict, dim=1)  # [B, L', N, D]
        predict = self.forecast_fc(predict)  # -> fk_dim
        return predict


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args.get("adjs", [])
        self.localized_st_conv = STLocalizedConv(
            hidden_dim,
            pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre,
            dy_graph=dy_graph,
            sta_graph=sta_graph,
            **model_args,
        )
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])
        self.forecast_branch = DifForecast(hidden_dim, fk_dim=fk_dim, **model_args)
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, X_spa, dynamic_graph, static_graph):
        Z = self.localized_st_conv(X_spa, dynamic_graph, static_graph)
        forecast_hidden = self.forecast_branch(X_spa, Z, self.localized_st_conv, dynamic_graph, static_graph)
        backcast_seq = self.backcast_branch(Z)
        X = X[:, -backcast_seq.shape[1] :, :, :]
        backcast_seq_res = self.residual_decompose(X, backcast_seq)
        return backcast_seq_res, forecast_hidden


# ===================== Dynamic graph constructor（简化版） =====================

class DistanceFunction(nn.Module):
    """简化版距离函数：只用节点 embedding 构建一张相似度图。"""

    def __init__(self, **model_args):
        super().__init__()

    def forward(self, X, E_d, E_u, T_D, D_W):
        # E_d, E_u: [N, D]
        sim = torch.relu(torch.matmul(E_d, E_u.T))  # [N, N]
        return [sim]


class Mask(nn.Module):
    """这里不做复杂的 top-k mask，只原样返回。"""

    def __init__(self, **model_args):
        super().__init__()

    def forward(self, graphs):
        return graphs


class Normalizer(nn.Module):
    """按行归一化 A -> D^{-1} A。"""

    def __init__(self):
        super().__init__()

    def forward(self, graphs):
        out = []
        for g in graphs:
            g = g + torch.eye(g.shape[0], device=g.device)
            g = g / (g.sum(-1, keepdim=True) + 1e-6)
            out.append(g)
        return out


class MultiOrder(nn.Module):
    """生成多阶邻接：A, A^2, ..., A^K。"""

    def __init__(self, order: int):
        super().__init__()
        self.order = order

    def forward(self, graphs):
        res = []
        for g in graphs:
            cur = []
            g_power = g
            for _ in range(self.order):
                cur.append(g_power)
                g_power = torch.matmul(g_power, g)
            res.append(cur)
        return res


class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args["k_s"]
        self.k_t = model_args["k_t"]
        self.hidden_dim = model_args["num_hidden"]
        self.node_dim = model_args["node_hidden"]

        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                # [N, N] -> [N, N, k_t, N] -> [N, N, k_t*N]
                k_order_graph = k_order_graph.unsqueeze(-2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0],
                    k_order_graph.shape[1],
                    k_order_graph.shape[2] * k_order_graph.shape[3],
                )
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, **inputs):
        X = inputs["X"]
        E_d = inputs["E_d"]
        E_u = inputs["E_u"]
        T_D = inputs["T_D"]
        D_W = inputs["D_W"]

        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)
        return dynamic_graphs


# ===================== Inherent block（时间解耦） =====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, X):
        X = X + self.pe[: X.size(0)]
        return self.dropout(X)


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # X: [B, L, N, D]
        B, L, N, D = X.shape
        X = X.transpose(1, 2).reshape(B * N, L, D)
        hx = torch.zeros_like(X[:, 0, :])
        outputs = []
        for t in range(L):
            hx = self.gru_cell(X[:, t, :], hx)
            outputs.append(hx)
        outputs = torch.stack(outputs, dim=0)  # [L, B*N, D]
        outputs = self.dropout(outputs)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        Z, _ = self.mha(X, K, V)
        return self.dropout(Z)


class InhForecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args["seq_length"]
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        B, L, N, D = X.shape
        Lh, BN, D = RNN_H.shape
        Lz, BN, D = Z.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(int(self.output_seq_len / self.model_args["gap"]) - 1):
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)  # [L', B*N, D]
        predict = predict.reshape(-1, B, N, D).transpose(0, 1)  # [B, L', N, D]
        predict = self.forecast_fc(predict)
        return predict


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, fk_dim=256, first=None, **model_args):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(hidden_dim, model_args["dropout"]) if first else None
        self.rnn_layer = RNNLayer(hidden_dim, model_args["dropout"])
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, model_args["dropout"], bias)
        self.forecast_block = InhForecast(hidden_dim, fk_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.sub_and_norm = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, X):
        B, L, N, D = X.shape
        RNN_H_raw = self.rnn_layer(X)  # [L, B*N, D]
        RNN_H = self.pos_encoder(RNN_H_raw) if self.pos_encoder is not None else RNN_H_raw
        Z = self.transformer_layer(RNN_H, RNN_H, RNN_H)  # [L, B*N, D]

        forecast_hidden = self.forecast_block(
            X, RNN_H_raw, Z, self.transformer_layer, self.rnn_layer, self.pos_encoder
        )

        Z = Z.reshape(L, B, N, D).transpose(0, 1)  # [B, L, N, D]
        backcast_seq = self.backcast_fc(Z)
        backcast_seq_res = self.sub_and_norm(X, backcast_seq)
        return backcast_seq_res, forecast_hidden


# ===================== Decouple layer & 主模型 =====================

class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, first=False, **model_args):
        super().__init__()
        self.spatial_gate = EstimationGate(
            model_args["node_hidden"], model_args["time_emb_dim"], 64, model_args["seq_length"]
        )
        self.dif_layer = DifBlock(hidden_dim, fk_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, fk_dim=fk_dim, first=first, **model_args)

    def forward(self, X, dynamic_graph, static_graph, E_u, E_d, T_D, D_W):
        # X: [B, L, N, D]
        X_spa = self.spatial_gate(E_u, E_d, T_D, D_W, X)
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            X=X, X_spa=X_spa, dynamic_graph=dynamic_graph, static_graph=static_graph
        )
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res)
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    """
    基于 BasicTS v0.5.8 的 D2STGNN 主体（做了轻微修改以适配你目前的数据格式，没有显式 time-of-day / day-of-week 通道）
    """

    def __init__(self, **model_args):
        super().__init__()
        self._in_feat = model_args["num_feat"]          # 输入特征维度 (= 55)
        self._hidden_dim = model_args["num_hidden"]     # 时空层 hidden dim
        self._node_dim = model_args["node_hidden"]
        self._forecast_dim = 256
        self._output_hidden = 512
        self._output_dim = model_args["seq_length"]     # 预测步数 H

        self._num_nodes = model_args["num_nodes"]
        self._k_s = model_args["k_s"]
        self._k_t = model_args["k_t"]
        self._num_layers = 5
        self._time_in_day_size = model_args["time_in_day_size"]
        self._day_in_week_size = model_args["day_in_week_size"]

        # 官方默认：不用预定义图，只用动态+静态自适应
        # 你如果想强制用 adj，可把 use_pre 改成 True
        model_args["use_pre"] = False
        model_args["dy_graph"] = False
        model_args["sta_graph"] = True
        self._model_args = model_args

        # 起始线性 embedding
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        # 时间 embedding（这里只是占位，用常数向量代替）
        self.T_i_D_emb = nn.Parameter(torch.empty(288, model_args["time_emb_dim"]))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args["time_emb_dim"]))

        # decouple 层
        self.layers = nn.ModuleList(
            [DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, first=True, **model_args)]
        )
        for _ in range(self._num_layers - 1):
            self.layers.append(DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args))

        # 动静态图构造
        if model_args["dy_graph"]:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)
        else:
            self.dynamic_graph_constructor = None

        # 可学习节点 embedding
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))

        # 输出回归层
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args["gap"])

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs['E_d']
        E_u = inputs['E_u']

    # 静态自适应图：由 node embedding 学出来
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []

    # 动态图：我们已经在 __init__ 里关掉了，这里就直接返回空列表
        if self._model_args['dy_graph'] and self.dynamic_graph_constructor is not None:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []

        return static_graph, dynamic_graph

    def _prepare_inputs(self, X):
        """
        这里做了一点改动：
        - 官方版本要求 X[..., num_feat] / num_feat+1 是 time_of_day & day_of_week，
          但你的数据只有 55 维数值特征，所以这里直接用一个常数时间 embedding。
        """
        num_feat = self._model_args["num_feat"]
        B, L, N, C = X.shape

        # 节点 embedding
        node_emb_u = self.node_emb_u  # [N, d]
        node_emb_d = self.node_emb_d  # [N, d]

        # 伪造时间 embedding：用第 0 个向量 broadcast
        T_vec = self.T_i_D_emb[0].view(1, 1, 1, -1)
        D_vec = self.D_i_W_emb[0].view(1, 1, 1, -1)
        T_i_D = T_vec.expand(B, L, N, -1)
        D_i_W = D_vec.expand(B, L, N, -1)

        # 真实信号
        X_sig = X[..., :num_feat]

        return X_sig, node_emb_u, node_emb_d, T_i_D, D_i_W

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        history_data: [B, L, N, C]  这里 C=55（SCADA + CERRA）
        返回: [B, H, N, 1]
        """
        X = history_data

        # 1) 准备节点 & 时间 embedding
        X, E_u, E_d, T_D, D_W = self._prepare_inputs(X)

        # 2) 构图
        static_graph, dynamic_graph = self._graph_constructor(E_u=E_u, E_d=E_d, X=X, T_D=T_D, D_W=D_W)

        # 3) 起始线性映射
        X = self.embedding(X)  # [B, L, N, hidden_dim]

        spa_forecast_hidden_list = []
        tem_forecast_hidden_list = []

        tem_backcast_seq_res = X
        for layer in self.layers:
            tem_backcast_seq_res, spa_f, tem_f = layer(
                tem_backcast_seq_res, dynamic_graph, static_graph, E_u, E_d, T_D, D_W
            )
            spa_forecast_hidden_list.append(spa_f)
            tem_forecast_hidden_list.append(tem_f)

        # 4) 输出层
        spa_forecast_hidden = sum(spa_forecast_hidden_list)
        tem_forecast_hidden = sum(tem_forecast_hidden_list)
        forecast_hidden = spa_forecast_hidden + tem_forecast_hidden  # [B, L', N, fk_dim]

        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))  # [B, L', N, gap]
        forecast = forecast.transpose(1, 2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        forecast = forecast.transpose(1, 2).unsqueeze(-1)  # [B, H, N, 1]
        return forecast


# ===================== 你在训练脚本中真正调用的包装类 =====================

class D2STGNN_v058(D2STGNN):
    """
    对外包装一层，方便在训练脚本里以更直观的参数初始化。
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        seq_len: int,
        horizon: int,
        adjs,
        k_s: int = 3,
        k_t: int = 3,
        node_hidden: int = 16,
        num_hidden: int = 64,
        time_emb_dim: int = 16,
        gap: int = 1,
        dropout: float = 0.3,
    ):
        model_args = dict(
            num_feat=input_dim,
            num_hidden=num_hidden,
            node_hidden=node_hidden,
            seq_length=horizon,
            num_nodes=num_nodes,
            k_s=k_s,
            k_t=k_t,
            adjs=adjs,
            time_in_day_size=288,
            day_in_week_size=7,
            time_emb_dim=time_emb_dim,
            gap=gap,
            dropout=dropout,
        )
        super().__init__(**model_args)