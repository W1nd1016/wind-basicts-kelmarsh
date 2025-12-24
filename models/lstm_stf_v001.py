# models/lstm_stf_v001.py
import torch
import torch.nn as nn


class LSTM_STF_v001(nn.Module):
    """
    Simple LSTM for Spatio-Temporal Forecasting (SCADA-only):
      - Input x:  [B, L, N, F_full]  （例如 F_full = 55）
      - 只使用前 scada_dim=7 维特征 -> [P, dP, W, dir_sin, dir_cos, nac_sin, nac_cos]
      - Output y_hat: [B, H, N]
    """

    def __init__(
        self,
        num_nodes: int,
        full_input_dim: int,   # 全部特征维度（例如 55）
        horizon: int,
        scada_dim: int = 7,    # 只用前 7 维 SCADA
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.full_input_dim = full_input_dim
        self.scada_dim = scada_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_layers = num_layers

        # LSTM 只吃 SCADA 特征：input_size = scada_dim
        self.lstm = nn.LSTM(
            input_size=scada_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 最后时刻 hidden -> H 步预测
        self.proj = nn.Linear(hidden_dim, horizon)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, N, F_full]
        return: [B, H, N]
        """
        B, L, N, F = x.shape
        assert N == self.num_nodes, f"num_nodes mismatch: x has {N}, model expects {self.num_nodes}"
        assert (
            F == self.full_input_dim
        ), f"input_dim mismatch: x has {F}, model expects {self.full_input_dim}"

        # 只取前 scada_dim 个特征：SCADA 部分
        # 对应你的 feature_names 开头 7 项:
        # ["P", "dP", "W", "dir_sin", "dir_cos", "nac_sin", "nac_cos"]
        x_scada = x[..., : self.scada_dim]     # [B, L, N, scada_dim]

        # (B, L, N, scada_dim) -> (B*N, L, scada_dim)
        x_reshaped = x_scada.view(B * N, L, self.scada_dim)

        # LSTM 时间建模
        out, (h_n, c_n) = self.lstm(x_reshaped)   # out: [B*N, L, hidden_dim]
        h_last = out[:, -1, :]                    # [B*N, hidden_dim]
        h_last = self.norm(h_last)

        # 预测 H 步: [B*N, H]
        y_hat = self.proj(h_last)

        # [B*N, H] -> [B, N, H] -> [B, H, N]
        y_hat = y_hat.view(B, N, self.horizon)
        y_hat = y_hat.permute(0, 2, 1)

        return y_hat