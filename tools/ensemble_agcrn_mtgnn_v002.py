# tools/ensemble_agcrn_mtgnn_v002.py
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
    """
    # 4D -> 3D (常见 [B, H, N, 1])
    if pred.dim() == 4:
        if pred.shape[-1] == 1:
            pred = pred[..., 0]
        else:
            raise RuntimeError(f"Unexpected 4D pred shape: {pred.shape}")

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
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        pred = model(x)
        pred = align_pred_shape(pred, y)

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
    全局统一权重：
        pred = w_ag * pred_agcrn + (1 - w_ag) * pred_mtgnn
    """
    agcrn.eval()
    mtgnn.eval()
    maes, rmses = [], []

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        pred_a = align_pred_shape(agcrn(x), y)
        pred_m = align_pred_shape(mtgnn(x), y)

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
    在验证集上暴力搜索全局最优权重 w_ag。
    """
    best_w = None
    best_mae = float('inf')
    best_rmse = None

    w = w_min
    ws = []
    maes = []
    while w <= w_max + 1e-9:
        mae, rmse = eval_ensemble(agcrn, mtgnn, val_loader, device, w_ag=w)
        ws.append(w)
        maes.append(mae)
        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            best_w = w
        w += step

    print("\n[VAL ENSEMBLE WEIGHT SEARCH - GLOBAL]")
    for w_i, mae_i in zip(ws, maes):
        print(f"  w_ag={w_i:.2f} -> val MAE={mae_i:.4f}")
    print(
        f"[VAL BEST GLOBAL] w_ag={best_w:.2f} | "
        f"val MAE={best_mae:.4f} | val RMSE={best_rmse:.4f}"
    )
    return best_w, best_mae, best_rmse


# ----------------- 按 horizon 单独调权重 -----------------
@torch.no_grad()
def eval_ensemble_single_h(agcrn, mtgnn, loader, device, w_ag, h_idx):
    """
    只评估第 h_idx 个 horizon 的 ensemble MAE / RMSE。
    """
    agcrn.eval()
    mtgnn.eval()
    maes, rmses = [], []

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        pred_a = align_pred_shape(agcrn(x), y)[:, h_idx:h_idx+1, :]  # [B, 1, N]
        pred_m = align_pred_shape(mtgnn(x), y)[:, h_idx:h_idx+1, :]

        y_h = y[:, h_idx:h_idx+1, :]
        m_h = m[:, h_idx:h_idx+1, :]

        pred_h = w_ag * pred_a + (1.0 - w_ag) * pred_m

        mae = masked_mae(pred_h, y_h, m_h).item()
        rmse = torch.sqrt(
            (m_h * (pred_h - y_h) ** 2).sum() / (m_h.sum() + 1e-6)
        ).item()
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


@torch.no_grad()
def search_best_weight_per_horizon(agcrn, mtgnn, val_loader, device,
                                   H, w_min=0.0, w_max=1.0, step=0.05):
    """
    对每一个 horizon h=0..H-1 分别搜索最优权重 w_ag^(h)。
    返回:
        best_ws: list[H]
    """
    best_ws = []
    print("\n[VAL ENSEMBLE WEIGHT SEARCH - PER HORIZON]")
    for h in range(H):
        best_w_h = None
        best_mae_h = float('inf')
        best_rmse_h = None

        w = w_min
        ws = []
        maes = []
        while w <= w_max + 1e-9:
            mae, rmse = eval_ensemble_single_h(agcrn, mtgnn, val_loader, device, w_ag=w, h_idx=h)
            ws.append(w)
            maes.append(mae)
            if mae < best_mae_h:
                best_mae_h = mae
                best_rmse_h = rmse
                best_w_h = w
            w += step

        best_ws.append(best_w_h)
        print(f"  Horizon {h} -> best w_ag={best_w_h:.2f}, "
              f"val MAE={best_mae_h:.4f}, val RMSE={best_rmse_h:.4f}")

    print(f"[VAL BEST PER-HORIZON WEIGHTS] {best_ws}")
    return best_ws


@torch.no_grad()
def eval_ensemble_per_h_weights(agcrn, mtgnn, loader, device, ws):
    """
    使用 horizon-wise 权重列表 ws[h] 做集成。
    """
    agcrn.eval()
    mtgnn.eval()
    maes, rmses = [], []

    H = len(ws)

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        pred_a = align_pred_shape(agcrn(x), y)  # [B, H, N]
        pred_m = align_pred_shape(mtgnn(x), y)  # [B, H, N]

        # 按 horizon 逐步融合
        pred = torch.zeros_like(y)
        for h in range(H):
            w_ag = ws[h]
            pred[:, h, :] = w_ag * pred_a[:, h, :] + (1.0 - w_ag) * pred_m[:, h, :]

        mae = masked_mae(pred, y, m).item()
        rmse = torch.sqrt(
            (m * (pred - y) ** 2).sum() / (m.sum() + 1e-6)
        ).item()
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


# ----------------- main -----------------
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

    # ===== 数据集 & DataLoader =====
    val_ds  = WindSTFDataset(split="val",  L=L, H=H)
    test_ds = WindSTFDataset(split="test", L=L, H=H)

    val_ld  = DataLoader(val_ds,  batch_size=128, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=128, shuffle=False)

    # ===== 构建 MTGNN_v058（对齐你训练脚本的超参） =====
    mtgnn = MTGNN_v058(
        num_nodes=num_nodes,
        in_dim=input_dim,   # 55
        seq_length=L,       # 24
        horizon=H,          # 6
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

    # ===== 构建 AGCRN（对齐 agcrn_min_best.pt 的超参） =====
    agcrn = AGCRN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        rnn_units=32,    # 你训练时使用的 hidden size
        horizon=H,
        num_layers=1,
        embed_dim=8,
        cheb_k=3,
    ).to(device)

    # ===== 加载 checkpoint =====
    mtgnn_ckpt = "output/mtgnn_v058_best.pt"
    agcrn_ckpt = "output/agcrn_min_best.pt"

    mtgnn.load_state_dict(torch.load(mtgnn_ckpt, map_location=device))
    print(f"Loaded MTGNN from {mtgnn_ckpt}")

    agcrn.load_state_dict(torch.load(agcrn_ckpt, map_location=device))
    print(f"Loaded AGCRN from {agcrn_ckpt}")

    # ===== 单模型 TEST 表现 =====
    agcrn_mae, agcrn_rmse = eval_model(agcrn, test_ld, device)
    print(f"\n[AGCRN   TEST] MAE {agcrn_mae:.4f} | RMSE {agcrn_rmse:.4f}")

    mtgnn_mae, mtgnn_rmse = eval_model(mtgnn, test_ld, device)
    print(f"[MTGNN   TEST] MAE {mtgnn_mae:.4f} | RMSE {mtgnn_rmse:.4f}")

    # ===== 1) 先做全局统一权重搜索（复现你上一版结果） =====
    best_w_global, _, _ = search_best_weight_on_val(
        agcrn, mtgnn, val_ld, device,
        w_min=0.0, w_max=1.0, step=0.05
    )
    ens_mae_g, ens_rmse_g = eval_ensemble(
        agcrn, mtgnn, test_ld, device, w_ag=best_w_global
    )
    print(
        f"\n[ENSEMBLE TEST - GLOBAL] "
        f"w_ag={best_w_global:.2f} | MAE {ens_mae_g:.4f} | RMSE {ens_rmse_g:.4f}"
    )

    # ===== 2) 再做 per-horizon 权重搜索 =====
    best_ws = search_best_weight_per_horizon(
        agcrn, mtgnn, val_ld, device,
        H=H, w_min=0.0, w_max=1.0, step=0.05
    )
    ens_mae_ph, ens_rmse_ph = eval_ensemble_per_h_weights(
        agcrn, mtgnn, test_ld, device, best_ws
    )
    print(
        f"\n[ENSEMBLE TEST - PER HORIZON] "
        f"ws={['%.2f' % w for w in best_ws]} | "
        f"MAE {ens_mae_ph:.4f} | RMSE {ens_rmse_ph:.4f}"
    )


if __name__ == "__main__":
    main()