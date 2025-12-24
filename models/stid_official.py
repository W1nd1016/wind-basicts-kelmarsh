# models/stid_official.py
import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links (官方 STID 里的 MLP)."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim,
            kernel_size=(1, 1), bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim,
            kernel_size=(1, 1), bias=True
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """input_data: [B, D, N, 1] -> [B, D, N, 1]"""

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data                                  # residual
        return hidden


class STIDCore(nn.Module):
    """
    官方 STID 主体（基本照抄 v0.5.8，只是类名改成 STIDCore，方便外面再包一层）。
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for
           Multivariate Time Series Forecasting (CIKM'22)
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes (和官方完全一致)
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer：把 L×C 展平成一个通道，再做 1×1 Conv
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        # encoding
        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )
        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layer)
            ]
        )

        # regression: 输出 horizon 个时间步
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None, # type: ignore
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        history_data: [B, L, N, C]
        return:      [B, output_len, N, 1]
        """

        # 只取前 input_dim 个特征（我们会让 input_dim = C）
        input_data = history_data[..., range(self.input_dim)]

        # ===== 时间 embedding（这里我们会关掉 if_T_i_D / if_D_i_W，所以默认不走）=====
        if self.if_time_in_day:
            # 注意：这里假设某个通道是归一化到 [0,1] 的 time_of_day
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            # 同理假设某个通道是归一化到 [0,1] 的 day_of_week
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)
            ]
        else:
            day_in_week_emb = None

        # ===== time series embedding =====
        batch_size, _, num_nodes, _ = input_data.shape
        # [B, L, N, C] -> [B, N, L*C] -> [B, L*C, N, 1]
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # [B, embed_dim, N, 1]
        time_series_emb = self.time_series_emb_layer(input_data)

        # ===== spatial embedding =====
        node_emb = []
        if self.if_spatial:
            # [1, N, D] -> [B, D, N, 1]
            node_emb.append(
                self.node_emb.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .transpose(1, 2)
                .unsqueeze(-1)
            )

        # ===== temporal embedding =====
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # ===== concat 所有 embedding: [B, hidden_dim, N, 1] =====
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression: [B, output_len, N, 1]
        prediction = self.regression_layer(hidden)

        return prediction


class STID(nn.Module):
    """
    外层包一层，把官方 STIDCore 接到你现在的 X->Y pipeline 上。
    输入: x [B, L, N, F]
    输出: pred [B, H, N]
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        input_len: int,
        horizon: int,
        # 一些可以调的超参:
        node_dim: int = 16,
        embed_dim: int = 64,
        num_layer: int = 3,
        if_node: bool = True,
        # 把时间 embedding 关掉，保持和你当前 X.npy 兼容
        if_T_i_D: bool = False,
        if_D_i_W: bool = False,
        temp_dim_tid: int = 16,
        temp_dim_diw: int = 16,
        time_of_day_size: int = 288,
        day_of_week_size: int = 7,
    ):
        super().__init__()
        model_args = dict(
            num_nodes=num_nodes,
            node_dim=node_dim,
            input_len=input_len,
            input_dim=input_dim,
            embed_dim=embed_dim,
            output_len=horizon,
            num_layer=num_layer,
            temp_dim_tid=temp_dim_tid,
            temp_dim_diw=temp_dim_diw,
            time_of_day_size=time_of_day_size,
            day_of_week_size=day_of_week_size,
            if_T_i_D=if_T_i_D,
            if_D_i_W=if_D_i_W,
            if_node=if_node,
        )
        self.core = STIDCore(**model_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, N, F]
        return: [B, H, N]
        """
        pred = self.core(
            history_data=x,
            future_data=None,
            batch_seen=0,
            epoch=0,
            train=self.training,
        )  # [B, H, N, 1]
        return pred.squeeze(-1)  # -> [B, H, N]