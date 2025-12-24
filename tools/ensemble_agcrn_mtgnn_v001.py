# tools/ensemble_agcrn_mtgnn_v001.py
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.wind_dataset import WindSTFDataset

from models.agcrn_bt import AGCRN
from models.mtgnn_v058 import MTGNN_v058

# ----------------- 通用指标函数 -----------------
def masked_mae(pred, y, m):
    """
    pred, y, m: [B, H, N]
    m: 1 表示有效点
    """
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)


def align_pred_shape(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    把任意奇怪维度的 pred 对齐成和 y 一样的 [B, H, N]
    y 的 shape 视为标准：
        y.shape = [B, H, N]

    常见几种情况：
        - [B, H, N]      -> 原样返回
        - [B, N, H]      -> permute(0, 2, 1)
        - [H, B, N]      -> permute(1, 0, 2)
        - [N, B, H]      -> permute(1, 2, 0)
        - 4D 情况（例如 [B, H, N, 1]）会先 squeeze 掉最后一维
    """
    # 先把可能的 4D 情况压成 3D
    if pred.dim() == 4:
        # 最常见：最后一维是长度 1
        if pred.shape[-1] == 1:
            pred = pred[..., 0]
        else:
            raise RuntimeError(f"Unexpected 4D pred shape: {pred.shape}")

    if pred.dim() != 3:
        raise RuntimeError(f"Pred must be 3D after squeeze, got {pred.shape}")

    B, H, N = y.shape

    # 完全一致，直接用
    if pred.shape == y.shape:
        return pred

    # [B, N, H] -> [B, H, N]
    if pred.shape == (B, N, H):
        return pred.permute(0, 2, 1)

    # [H, B, N] -> [B, H, N]
    if pred.shape == (H, B, N):
        return pred.permute(1, 0, 2)

    # [N, B, H] -> [B, H, N]
    if pred.shape == (N, B, H):
        return pred.permute(1, 2, 0)

    # 其它奇怪情况就直接报错，方便之后针对性改
    raise RuntimeError(
        f"Cannot align pred shape {pred.shape} to y shape {y.shape}"
    )


@torch.no_grad()
def eval_model(model, loader, device):
    """
    评估单个模型：
      - 自动将 model(x) 的输出对齐成 [B, H, N]
      - 计算 masked MAE / RMSE
    """
    model.eval()
    maes, rmses = [], []

    for x, y, m in loader:
        # x: [B, L, N, F]
        # y, m: [B, H, N]
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        pred = model(x)
        pred = align_pred_shape(pred, y)  # 对齐成 [B, H, N]

        mae = masked_mae(pred, y, m).item()
        rmse = torch.sqrt(
            (m * (pred - y) ** 2).sum() / (m.sum() + 1e-6)
        ).item()
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


@torch.no_grad()
def eval_ensemble(agcrn, mtgnn, loader, device, w_ag=0.5):
    """
    简单加权集成：
        pred = w_ag * pred_agcrn + (1 - w_ag) * pred_mtgnn
    """
    agcrn.eval()
    mtgnn.eval()
    maes, rmses = [], []

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        pred_a = agcrn(x)
        pred_m = mtgnn(x)

        pred_a = align_pred_shape(pred_a, y)
        pred_m = align_pred_shape(pred_m, y)

        pred = w_ag * pred_a + (1.0 - w_ag) * pred_m   # [B, H, N]

        mae = masked_mae(pred, y, m).item()
        rmse = torch.sqrt(
            (m * (pred - y) ** 2).sum() / (m.sum() + 1e-6)
        ).item()
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


@torch.no_grad()
def search_best_weight_on_val(agcrn, mtgnn, val_loader, device,
                              w_min=0.0, w_max=1.0, step=0.05):
    """
    在验证集上暴力搜索最优权重 w_ag：
        w_ag ∈ [w_min, w_max]，步长为 step
    返回:
        best_w, best_mae, best_rmse
    """
    best_w = None
    best_mae = float('inf')
    best_rmse = None

    w = w_min
    ws = []
    maes = []
    while w <= w_max + 1e-9:  # 避免浮点问题
        mae, rmse = eval_ensemble(agcrn, mtgnn, val_loader, device, w_ag=w)
        ws.append(w)
        maes.append(mae)
        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            best_w = w
        w += step

    print("\n[VAL ENSEMBLE WEIGHT SEARCH]")
    for w_i, mae_i in zip(ws, maes):
        print(f"  w_ag={w_i:.2f} -> val MAE={mae_i:.4f}")
    print(
        f"[VAL BEST] w_ag={best_w:.2f} | val MAE={best_mae:.4f} | val RMSE={best_rmse:.4f}"
    )
    return best_w, best_mae, best_rmse


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

    # ===== 加载邻接矩阵 =====
    try:
        adj = np.load("data/wind/adj.npy").astype(np.float32)
        print("Loaded adj.npy, shape:", adj.shape)
    except FileNotFoundError:
        print("WARNING: data/wind/adj.npy not found, use identity graph instead.")
        adj = np.eye(num_nodes, dtype=np.float32)
    adj_tensor = torch.tensor(adj, dtype=torch.float32)

    # ===== 数据集与 DataLoader =====
    val_ds  = WindSTFDataset(split="val",  L=L, H=H)
    test_ds = WindSTFDataset(split="test", L=L, H=H)

    val_ld  = DataLoader(val_ds,  batch_size=128, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=128, shuffle=False)

    # ===== 构建 MTGNN_v058（和你训练时保持一致） =====
    mtgnn = MTGNN_v058(
        num_nodes=num_nodes,
        in_dim=input_dim,   # 55
        seq_length=L,       # 24
        horizon=H,          # 6
        predefined_A=adj_tensor,
        gcn_true=True,
        buildA_true=False,  # 用静态 adj
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

    # ===== 构建 AGCRN（参数对齐你训练时使用的配置） =====
    agcrn = AGCRN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        rnn_units=32,    # 保持和训练 agcrn_min_best.pt 时一致
        horizon=H,
        num_layers=1,
        embed_dim=8,
        cheb_k=3,
    ).to(device)

    # ===== 加载已经训练好的权重 =====
    mtgnn_ckpt = "output/mtgnn_v058_best.pt"
    agcrn_ckpt = "output/agcrn_min_best.pt"

    # ---- 加载 MTGNN ----
    mtgnn.load_state_dict(torch.load(mtgnn_ckpt, map_location=device))
    print(f"Loaded MTGNN from {mtgnn_ckpt}")

    # ---- 加载 AGCRN ----
    agcrn.load_state_dict(torch.load(agcrn_ckpt, map_location=device))
    print(f"Loaded AGCRN from {agcrn_ckpt}")

    # ===== 单模型在 TEST 上的表现 =====
    agcrn_mae, agcrn_rmse = eval_model(agcrn, test_ld, device)
    print(f"\n[AGCRN   TEST] MAE {agcrn_mae:.4f} | RMSE {agcrn_rmse:.4f}")

    mtgnn_mae, mtgnn_rmse = eval_model(mtgnn, test_ld, device)
    print(f"[MTGNN   TEST] MAE {mtgnn_mae:.4f} | RMSE {mtgnn_rmse:.4f}")

    # ===== 在验证集上搜索最优权重 =====
    best_w, best_val_mae, best_val_rmse = search_best_weight_on_val(
        agcrn, mtgnn, val_ld, device,
        w_min=0.0, w_max=1.0, step=0.05
    )

    # ===== 用最优权重在 TEST 上做最终 ensemble =====
    ens_mae, ens_rmse = eval_ensemble(agcrn, mtgnn, test_ld, device, w_ag=best_w)
    print(
        f"\n[ENSEMBLE TEST] "
        f"w_ag={best_w:.2f} | MAE {ens_mae:.4f} | RMSE {ens_rmse:.4f}"
    )


if __name__ == "__main__":
    main()