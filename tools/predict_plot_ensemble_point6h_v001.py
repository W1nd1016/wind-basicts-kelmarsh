# tools/predict_plot_ensemble_point6h_v001.py
import os
import sys
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

# 让代码可以找到 datasets / models
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset import WindSTFDataset
from models.agcrn_bt import AGCRN
from models.mtgnn_v058 import MTGNN_v058


def align_pred_shape(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    把任意奇怪维度的 pred 对齐成和 y 一样的 [B, H, N]
    y 的 shape 视为标准：
        y.shape = [B, H, N]
    常见几种情况：
        - [B, H, N]      -> 原样返回
        - [B, N, H]      -> permute(0, 2, 1)
        - [H, B, N]      -> permute(1, 0, 2)
        - [B, H, N, 1]   -> squeeze 最后一维
    """
    # 处理 4 维情况，通常是 [B, H, N, 1]
    if pred.dim() == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]

    if pred.dim() != 3:
        raise RuntimeError(f"Pred must be 3D after squeeze, got {pred.shape}")

    B, H, N = y.shape

    if pred.shape == y.shape:
        return pred

    if pred.shape == (B, N, H):
        return pred.permute(0, 2, 1)

    if pred.shape == (H, B, N):
        return pred.permute(1, 0, 2)

    if pred.shape == (N, B, H):
        return pred.permute(1, 2, 0)

    raise RuntimeError(
        f"Cannot align pred shape {pred.shape} to y shape {y.shape}"
    )


def main():
    # ======= 可调参数：选一个测试样本 & 涡轮机 =======
    SAMPLE_IDX = 238   # 在 test 集中的样本下标，0 ~ len(test_ds)-1，可以自己改
    TURBINE_INDEX = 0  # 第几台风机，0~5
    W_AG = 0.4         # Ensemble 中 AGCRN 的权重（之前搜索得到的全局最优）

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ======= 读取 meta.json，拿到基本信息和反归一化参数 =======
    meta_path = "data/wind/meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    input_dim = len(feature_names)
    turbine_ids = meta["turbine_ids"]
    num_nodes = len(turbine_ids)

    L = meta.get("L", meta.get("history_len", 24))
    H = meta.get("H", meta.get("horizon", 6))

    y_mu = float(meta["y_mu"])
    y_sd = float(meta["y_sd"])
    x_mu = np.array(meta["x_mu"], dtype=np.float32)  # [F_total]
    x_sd = np.array(meta["x_sd"], dtype=np.float32)

    print("input_dim =", input_dim)
    print("num_nodes =", num_nodes)
    print("L =", L, "H =", H)
    print("turbine_ids =", turbine_ids)
    print("y_mu (kW) =", y_mu, "y_sd (kW) =", y_sd)

    # ======= 找出 feature 里 P 的下标，用来反归一化历史功率 =======
    assert "P" in feature_names, "feature_names 中没有 'P'"
    p_idx = feature_names.index("P")
    print("Index of 'P' in feature_names:", p_idx)

    # ======= 读取邻接矩阵，用于构建 MTGNN =======
    adj = np.load("data/wind/adj.npy").astype(np.float32)
    adj_tensor = torch.tensor(adj, dtype=torch.float32, device=device)
    print("Loaded adj.npy, shape:", adj.shape)

    # ======= 构建数据集，只用 test split =======
    test_ds = WindSTFDataset(split="test", L=L, H=H)
    print("len(test_ds) =", len(test_ds))
    assert 0 <= SAMPLE_IDX < len(test_ds), "SAMPLE_IDX 越界了，请改小一点"

    # 从 test 集中拿出一个样本（不通过 DataLoader）
    # x:[L,N,F], y:[H,N], m:[H,N]，全部都是“归一化后的特征/目标”
    x, y, m = test_ds[SAMPLE_IDX]

    # 增加 batch 维度 -> [1, L, N, F], [1, H, N], [1, H, N]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    m = m.unsqueeze(0).to(device)

    # ======= 构建 AGCRN & MTGNN（参数要跟训练时一致） =======
    agcrn = AGCRN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        rnn_units=32,   # 和你训练 agcrn_min_best.pt 时用的一致
        horizon=H,
        num_layers=1,
        embed_dim=8,
        cheb_k=3,
    ).to(device)

    mtgnn = MTGNN_v058(
        num_nodes=num_nodes,
        in_dim=input_dim,
        seq_length=L,
        horizon=H,
        predefined_A=adj_tensor,
        gcn_true=True,
        buildA_true=False,
        gcn_depth=2,
        dropout=0.3,
        subgraph_size=num_nodes,
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
    ).to(device)

    # ======= 加载已经训练好的权重 =======
    ag_ckpt = "output/agcrn_min_best.pt"
    mt_ckpt = "output/mtgnn_v058_best.pt"

    agcrn.load_state_dict(torch.load(ag_ckpt, map_location=device))
    mtgnn.load_state_dict(torch.load(mt_ckpt, map_location=device))
    print(f"Loaded AGCRN from {ag_ckpt}")
    print(f"Loaded MTGNN from {mt_ckpt}")

    agcrn.eval()
    mtgnn.eval()

    # ======= 单样本前向推理，得到 H=6 个小时的预测（归一化空间） =======
    with torch.no_grad():
        pred_ag = agcrn(x)   # 形状可能不是 [1,H,N]，下面对齐
        pred_mt = mtgnn(x)

        pred_ag = align_pred_shape(pred_ag, y)
        pred_mt = align_pred_shape(pred_mt, y)

        pred_ens = W_AG * pred_ag + (1.0 - W_AG) * pred_mt  # [1,H,N]

    # ======= 反归一化未来 6 小时的真实值和预测值（功率 kW） =======
    # y: [1,H,N]（标准化）
    y_true = y.cpu().numpy()[0] * y_sd + y_mu        # [H,N]
    ag_pred = pred_ag.cpu().numpy()[0] * y_sd + y_mu # [H,N]
    mt_pred = pred_mt.cpu().numpy()[0] * y_sd + y_mu # [H,N]
    ens_pred = pred_ens.cpu().numpy()[0] * y_sd + y_mu  # [H,N]

    # ======= 从输入 x 中恢复“前 24 小时历史真实功率” =======
    # x 现在是 [1,L,N,F]，是归一化后的特征；我们只取 P 这一维，然后反归一化：
    x_np = x.cpu().numpy()[0]        # [L,N,F]
    x_norm_p = x_np[:, :, p_idx]     # [L,N]，P 的归一化值
    P_hist = x_norm_p * x_sd[p_idx] + x_mu[p_idx]  # [L,N]，kW

    # 选择某一台风机
    n = TURBINE_INDEX
    assert 0 <= n < num_nodes, "TURBINE_INDEX 越界了"

    turbine_id = turbine_ids[n]

    hist_true_n = P_hist[:, n]   # [L]
    y_true_n = y_true[:, n]      # [H]
    ag_pred_n = ag_pred[:, n]
    mt_pred_n = mt_pred[:, n]
    ens_pred_n = ens_pred[:, n]

    # ======= 构造 x 轴：前 24 小时 + 后 6 小时 =======
    # 这里用“相对小时数”表示：
    #   -24, -23, ..., -1  表示历史 24 小时
    #    0,  1,  ...,  5  表示未来 6 个预测步（0 表示最近一小时的预测）
    hist_pos = np.arange(-L, 0)   # [-24, -23, ..., -1]
    fut_pos = np.arange(0, H)     # [0, 1, 2, 3, 4, 5]

    # ======= 画图 =======
    plt.figure(figsize=(8, 4))

    # 历史真实功率：只用一条线（蓝色实线）
    plt.plot(hist_pos, hist_true_n, label="History True (kW)", color="C0")

    # 未来真实功率：蓝色虚线，接在后面
    plt.plot(fut_pos, y_true_n, label="Future True (kW)", color="C0", linestyle="--")

    # 未来预测：AGCRN / MTGNN / Ensemble
    plt.plot(fut_pos, ag_pred_n, marker="o", label="AGCRN (kW)")
    plt.plot(fut_pos, mt_pred_n, marker="o", label="MTGNN (kW)")
    plt.plot(
        fut_pos,
        ens_pred_n,
        marker="o",
        label=f"Ensemble (w_ag={W_AG:.2f}) (kW)",
    )

    # 在 0 处画一条竖线，分隔“输入窗口”和“预测窗口”
    plt.axvline(0, color="k", linestyle="--", alpha=0.7)
    plt.text(
        -L + 1,
        np.max(np.concatenate([hist_true_n, y_true_n, ens_pred_n])) * 1.02,
        "History 24h",
        fontsize=9,
        ha="left",
    )
    plt.text(
        0.1,
        np.max(np.concatenate([hist_true_n, y_true_n, ens_pred_n])) * 1.02,
        "Forecast 6h",
        fontsize=9,
        ha="left",
    )

    plt.xlabel("Hours offset (0 = forecast start)")
    plt.ylabel("Power (kW)")
    plt.title(
        f"Turbine {turbine_id} | test sample #{SAMPLE_IDX}\n"
        f"History 24h + Forecast 6h"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs("figs", exist_ok=True)
    out_path = f"figs/point6h_with_history_turb{turbine_id}_idx{SAMPLE_IDX}.png"
    plt.savefig(out_path, dpi=150)
    print("Saved figure to", out_path)


if __name__ == "__main__":
    main()