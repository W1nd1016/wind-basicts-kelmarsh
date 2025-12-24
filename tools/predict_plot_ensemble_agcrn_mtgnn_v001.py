# tools/predict_plot_ensemble_agcrn_mtgnn_v002.py
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.wind_dataset import WindSTFDataset
from models.agcrn_bt import AGCRN
from models.mtgnn_v058 import MTGNN_v058

# =============== 一些超参 / 可调参数 ===============
W_AG_GLOBAL = 0.40   # 全局集成权重 w_ag（你之前搜索出来的最佳 0.40）

# 时间序列可视化的“窗口长度”和起点：
PLOT_LEN   = 48     # 一次只看 200 个时间步（你可以改成 100, 300 等）
PLOT_START = None    # 如果为 None 就默认看“最后 PLOT_LEN 个时间步”
# 你也可以手动指定，比如 PLOT_START = 0（看一开始），或 500 等

# 想看哪台机组 & 哪个 horizon
PLOT_TURBINE_ID = 0  # 0~5
PLOT_HORIZON    = 5  # 0~5 (第几个预测步)


# ----------------- 工具函数 -----------------
def align_pred_shape(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    把 pred 对齐成和 y 一样的 [B, H, N] 形状。
    y.shape = [B, H, N] 视为标准。
    """
    # 有些模型可能输出 [B, H, N, 1]，先 squeeze 掉：
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

    raise RuntimeError(f"Cannot align pred shape {pred.shape} to y shape {y.shape}")


def masked_mae(pred, y, m):
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)


@torch.no_grad()
def collect_predictions(agcrn, mtgnn, loader, device, w_ag=0.4):
    """
    遍历 test loader，收集：
      - y_true: [S, H, N]
      - y_pred_ens: [S, H, N]
    S = 测试集中样本个数（滑窗起点数）
    """
    agcrn.eval()
    mtgnn.eval()

    all_true = []
    all_pred_ens = []

    for x, y, m in loader:
        # x: [B, L, N, F]
        # y, m: [B, H, N]（归一化后的）
        x = x.to(device)
        y = y.to(device)

        # 单模型预测
        pred_a = agcrn(x)
        pred_m = mtgnn(x)

        pred_a = align_pred_shape(pred_a, y)
        pred_m = align_pred_shape(pred_m, y)

        # 集成
        pred_ens = w_ag * pred_a + (1.0 - w_ag) * pred_m

        all_true.append(y.cpu().numpy())
        all_pred_ens.append(pred_ens.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)      # [S, H, N]
    all_pred_ens = np.concatenate(all_pred_ens, axis=0)  # [S, H, N]
    return all_true, all_pred_ens


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== 读取 meta.json =====
    meta_path = "data/wind/meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    input_dim = len(feature_names)
    print("input_dim =", input_dim)
    print("features =", feature_names)

    turbine_ids = meta.get("turbine_ids", None)
    if turbine_ids is not None:
        num_nodes = len(turbine_ids)
    else:
        num_nodes = meta.get("num_nodes", 6)
    print("num_nodes =", num_nodes)

    L = meta.get("L", meta.get("history_len", 24))
    H = meta.get("H", meta.get("horizon", 6))
    print("L =", L, "H =", H)

    # 反归一化用的参数
    y_mu = meta["y_mu"]
    y_sd = meta["y_sd"]
    print("y_mu =", y_mu, "y_sd =", y_sd)

    # ===== 邻接矩阵（给 MTGNN 用） =====
    adj = np.load("data/wind/adj.npy").astype(np.float32)
    print("Loaded adj.npy, shape:", adj.shape)
    adj_tensor = torch.tensor(adj, dtype=torch.float32, device=device)

    # ===== 数据集 / loader =====
    test_ds = WindSTFDataset(split="test", L=L, H=H)
    test_ld = DataLoader(test_ds, batch_size=128, shuffle=False)

    # ===== 构建模型并加载权重 =====
    # ---- MTGNN ----
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

    mtgnn_ckpt = "output/mtgnn_v058_best.pt"
    mtgnn.load_state_dict(torch.load(mtgnn_ckpt, map_location=device))
    print(f"Loaded MTGNN from {mtgnn_ckpt}")

    # ---- AGCRN ----
    agcrn = AGCRN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        rnn_units=32,
        horizon=H,
        num_layers=1,
        embed_dim=8,
        cheb_k=3,
    ).to(device)

    agcrn_ckpt = "output/agcrn_min_best.pt"
    agcrn.load_state_dict(torch.load(agcrn_ckpt, map_location=device))
    print(f"Loaded AGCRN from {agcrn_ckpt}")

    # ===== 收集测试集上的预测（归一化单位） =====
    y_true_norm, y_pred_norm = collect_predictions(
        agcrn, mtgnn, test_ld, device, w_ag=W_AG_GLOBAL
    )
    # y_true_norm, y_pred_norm: [S, H, N]

    # ===== 反归一化到功率单位（kW） =====
    # y_mu, y_sd 是标量
    y_true = y_true_norm * y_sd + y_mu    # [S, H, N]
    y_pred = y_pred_norm * y_sd + y_mu    # [S, H, N]

    S, H_, N = y_true.shape
    print(f"Collected predictions: S={S}, H={H_}, N={N}")

    # ========= 1. 时间序列图（带短窗口） =========
    h = PLOT_HORIZON
    n = PLOT_TURBINE_ID

    series_true = y_true[:, h, n]   # [S]
    series_pred = y_pred[:, h, n]   # [S]

    # 确定要画的时间窗口
    if PLOT_START is None:
        # 默认画最后 PLOT_LEN 个时间步
        end = S
        start = max(0, S - PLOT_LEN)
    else:
        start = max(0, PLOT_START)
        end = min(S, start + PLOT_LEN)

    print(f"Time series plot range: start={start}, end={end} (len={end-start})")

    t_axis = np.arange(start, end)

    plt.figure(figsize=(10, 4))
    plt.plot(t_axis, series_true[start:end], label="True (kW)", linewidth=1.5)
    plt.plot(t_axis, series_pred[start:end], label="Ensemble Pred (kW)", linewidth=1.2, alpha=0.8)
    plt.xlabel("Test sample index (sliding window index)")
    plt.ylabel("Power (kW)")
    plt.title(f"Turbine {n}, Horizon {h}, window [{start}, {end})")
    plt.legend()
    plt.tight_layout()
    os.makedirs("output/plots", exist_ok=True)
    ts_path = f"output/plots/ts_t{n}_h{h}_start{start}_len{end-start}.png"
    plt.savefig(ts_path, dpi=150)
    plt.close()
    print("Saved time series plot to:", ts_path)

    # ========= 2. 散点图（全测试集） =========
    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)

    plt.figure(figsize=(5, 5))
    plt.scatter(true_flat, pred_flat, s=5, alpha=0.4)
    max_val = max(true_flat.max(), pred_flat.max())
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=1)  # y=x 对角线
    plt.xlabel("True Power (kW)")
    plt.ylabel("Pred Power (kW)")
    plt.title("Ensemble: True vs Pred (all turbines, all horizons)")
    plt.tight_layout()
    scatter_path = "output/plots/scatter_true_vs_pred_ensemble.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print("Saved scatter plot to:", scatter_path)


if __name__ == "__main__":
    main()