# models/mtgnn_v058.py
# 自包含 MTGNN + Fan 专用 MTGNN_v058 封装版
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# =========================
# 基础图卷积 /传播模块
# =========================
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: [B, C, V, T], A: [V, V]
        # 输出: [B, C, V, T]
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        # x: [B, C, V, T], A: [B, V, W, ?] 动态图，这里用不上
        x = torch.einsum("ncvl,nvwl->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=bias,
        )

    def forward(self, x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        # x: [B, C, V, T], adj: [V, V]
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        a = adj / d.view(-1, 1)
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        # x: [B, C, V, T], adj: [V, V]
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)  # 通道维拼接
        ho = self.mlp(ho)
        return ho


class dy_mixprop(nn.Module):
    # 这个在本项目中暂时没用到，保留以防万一
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        self.mlp2 = linear((gdep + 1) * c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)

    def forward(self, x):
        # x: [B, C, V, T]
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

        # 正向传播
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        # 反向传播
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1 + ho2


# =========================
# 时间卷积模块
# =========================
class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout_each = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(
                nn.Conv2d(cin, cout_each, (1, kern), dilation=(1, dilation_factor))
            )

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        # 对齐时间维度
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3) :]
        x = torch.cat(x, dim=1)
        return x


# =========================
# 图构造模块
# =========================
class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        # idx: [V]
        if self.static_feat is None:
            idx = idx.to(self.emb1.weight.device)
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            idx = idx.to(self.static_feat.device)
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(adj.device)
        mask.fill_(float(0))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes), requires_grad=True)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0))
        mask.fill_(float(0))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0))
        mask.fill_(float(0))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


# =========================
# LayerNorm 替换实现
# =========================
class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        # input: [B, C, N, T]
        if self.elementwise_affine:
            return F.layer_norm(
                input,
                tuple(input.shape[1:]),
                self.weight[:, idx, :],
                self.bias[:, idx, :],
                self.eps,
            )
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


# =========================
# 官方 MTGNN 主体
# =========================
class MTGNN(nn.Module):
    """
    Paper: Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks
    Ref Official Code: https://github.com/nnzhan/MTGNN
    """

    def __init__(
        self,
        gcn_true,
        buildA_true,
        gcn_depth,
        num_nodes,
        predefined_A=None,
        static_feat=None,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        seq_length=12,
        in_dim=2,
        out_dim=12,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
    ):
        super(MTGNN, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = (
            predefined_A if predefined_A is not None else None
        )  # tensor [V, V] or None

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )

        self.gc = graph_constructor(
            num_nodes, subgraph_size, node_dim, alpha=tanhalpha, static_feat=static_feat
        )

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential**layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1
                    + i
                    * (kernel_size - 1)
                    * (dilation_exponential**layers - 1)
                    / (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1

            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i
                        + (kernel_size - 1)
                        * (dilation_exponential**j - 1)
                        / (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.gate_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.seq_length - rf_size_j + 1),
                        )
                    )
                else:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.receptive_field - rf_size_j + 1),
                        )
                    )

                if self.gcn_true:
                    self.gconv1.append(
                        mixprop(
                            conv_channels, residual_channels, gcn_depth, dropout, propalpha
                        )
                    )
                    self.gconv2.append(
                        mixprop(
                            conv_channels, residual_channels, gcn_depth, dropout, propalpha
                        )
                    )
                else:
                    # 不使用图卷积时走 1x1 卷积
                    pass

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm(
                            (residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline,
                        )
                    )
                else:
                    self.norm.append(
                        LayerNorm(
                            (residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline,
                        )
                    )

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True,
            )
        else:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self.idx = torch.arange(self.num_nodes)

    def forward(self, history_data: torch.Tensor, idx: int = None, **kwargs) -> torch.Tensor:
        """
        Args:
            history_data: [B, L, N, C]
        Returns:
            x: [B, out_dim, N, 1]
        """
        # [B, L, N, C] -> [B, C, N, L]
        history_data = history_data.transpose(1, 3).contiguous()
        seq_len = history_data.size(3)
        assert (
            seq_len == self.seq_length
        ), f"input sequence length {seq_len} != preset {self.seq_length}"

        # padding 以满足感受野
        if self.seq_length < self.receptive_field:
            history_data = F.pad(
                history_data, (self.receptive_field - self.seq_length, 0, 0, 0)
            )

        # 图结构
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx.to(history_data.device))
                else:
                    adp = self.gc(idx.to(history_data.device))
            else:
                adp = self.predefined_A.to(history_data.device)
        else:
            adp = None

        x = self.start_conv(history_data)
        skip = self.skip0(F.dropout(history_data, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter_out = self.filter_convs[i](x)
            filter_out = torch.tanh(filter_out)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter_out * gate
            x = F.dropout(x, self.dropout, training=self.training)

            s = self.skip_convs[i](x)
            skip = s + skip

            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]

            if idx is None:
                x = self.norm[i](x, self.idx.to(history_data.device))
            else:
                x = self.norm[i](x, idx.to(history_data.device))

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [B, out_dim, N, 1]

        return x


# =========================
# Fan 专用封装：MTGNN_v058
# =========================
class MTGNN_v058(nn.Module):
    """
    Wrapper for MTGNN to fit Fan's wind pipeline:
    - Input:  history_data [B, L, N, F]  (F=55, 已包含 SCADA + CERRA)
    - Output: pred [B, H, N] (归一化空间)
    """

    def __init__(
        self,
        num_nodes,
        in_dim,
        seq_length,
        horizon,
        predefined_A,
        gcn_true=True,
        buildA_true=False,  # 使用静态 adj，不学习 A
        gcn_depth=2,
        dropout=0.3,
        subgraph_size=None,
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        layers=3,
        propalpha=0.05,
        tanhalpha=3.0,
        layer_norm_affine=True,
    ):
        super().__init__()

        if subgraph_size is None:
            subgraph_size = num_nodes

        # 把邻接矩阵变成 float32 tensor
        if isinstance(predefined_A, torch.Tensor):
            A = predefined_A.float()
        else:
            A = torch.tensor(predefined_A, dtype=torch.float32)

        self.backbone = MTGNN(
            gcn_true=gcn_true,
            buildA_true=buildA_true,
            gcn_depth=gcn_depth,
            num_nodes=num_nodes,
            predefined_A=A,
            static_feat=None,
            dropout=dropout,
            subgraph_size=subgraph_size,
            node_dim=node_dim,
            dilation_exponential=dilation_exponential,
            conv_channels=conv_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            seq_length=seq_length,
            in_dim=in_dim,
            out_dim=horizon,  # 预测 H 个步长
            layers=layers,
            propalpha=propalpha,
            tanhalpha=tanhalpha,
            layer_norm_affline=layer_norm_affine,
        )

        self.num_nodes = num_nodes
        self.horizon = horizon
        self.seq_length = seq_length
        self.in_dim = in_dim

    def forward(self, history_data, idx=None):
        """
        history_data: [B, L, N, F]
        return: [B, H, N]
        """
        # 确保邻接矩阵在同一个 device 上
        if self.backbone.gcn_true and (not self.backbone.buildA_true):
            self.backbone.predefined_A = self.backbone.predefined_A.to(
                history_data.device
            )

        out4d = self.backbone(history_data, idx=idx)  # [B, H, N, 1]
        pred = out4d.squeeze(-1)  # -> [B, H, N]
        return pred