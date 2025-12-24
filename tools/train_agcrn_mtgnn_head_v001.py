# tools/train_agcrn_mtgnn_head_v001.py
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.wind_dataset import WindSTFDataset
from models.agcrn_bt import AGCRN
from models.mtgnn_v058 import MTGNN_v058


# ----------------- 通用函数 -----------------
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
    评估单个 base model（AGCRN 或 MTGNN）
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


# ----------------- 融合 Head -----------------
class AGCRN_MTGNNEensembleHead(nn.Module):
    """
    输入：AGCRN & MTGNN 的预测 [B, H, N]
    输出：融合后的预测 [B, H, N]

    升级版：2 -> 8 -> 1 的 1x1 Conv2d MLP，非线性融合
    """
    def __init__(self, hidden_channels: int = 8):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1),
        )

    def forward(self, pred_a: torch.Tensor, pred_m: torch.Tensor) -> torch.Tensor:
        # pred_a, pred_m: [B, H, N]
        x = torch.stack([pred_a, pred_m], dim=1)  # [B, 2, H, N]
        out = self.fuse(x)                        # [B, 1, H, N]
        return out[:, 0]                          # [B, H, N]

@torch.no_grad()
def eval_fusion_head(agcrn, mtgnn, head, loader, device):
    """
    评估融合 head（AGCRN + MTGNN + head）
    """
    agcrn.eval()
    mtgnn.eval()
    head.eval()

    maes, rmses = [], []

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        # base model 只做前向即可
        pred_a = align_pred_shape(agcrn(x), y)  # [B, H, N]
        pred_m = align_pred_shape(mtgnn(x), y)  # [B, H, N]

        pred = head(pred_a, pred_m)            # [B, H, N]

        mae = masked_mae(pred, y, m).item()
        rmse = torch.sqrt(
            (m * (pred - y) ** 2).sum() / (m.sum() + 1e-6)
        ).item()
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


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

    # ===== 邻接矩阵 =====
    try:
        adj = np.load("data/wind/adj.npy").astype(np.float32)
        print("Loaded adj.npy, shape:", adj.shape)
    except FileNotFoundError:
        print("WARNING: data/wind/adj.npy not found, use identity graph instead.")
        adj = np.eye(num_nodes, dtype=np.float32)
    adj_tensor = torch.tensor(adj, dtype=torch.float32)

    # ===== 数据集 & DataLoader =====
    train_ds = WindSTFDataset(split="train", L=L, H=H)
    val_ds   = WindSTFDataset(split="val",   L=L, H=H)
    test_ds  = WindSTFDataset(split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    # ===== 构建 MTGNN_v058（参数对齐你之前训练脚本） =====
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

    # ===== 构建 AGCRN（参数对齐 agcrn_min_best.pt） =====
    agcrn = AGCRN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        rnn_units=32,
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

    # ===== 冻结 backbone =====
    for p in agcrn.parameters():
        p.requires_grad = False
    for p in mtgnn.parameters():
        p.requires_grad = False

    agcrn.eval()
    mtgnn.eval()

    # ===== baseline：单模型 TEST 表现 =====
    ag_mae, ag_rmse = eval_model(agcrn, test_ld, device)
    mt_mae, mt_rmse = eval_model(mtgnn, test_ld, device)
    print(f"\n[BASELINE AGCRN TEST] MAE {ag_mae:.4f} | RMSE {ag_rmse:.4f}")
    print(f"[BASELINE MTGNN TEST] MAE {mt_mae:.4f} | RMSE {mt_rmse:.4f}")

    # ===== 构建融合 head =====
    head = AGCRN_MTGNNEensembleHead(hidden_channels=8).to(device)

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=3e-3,         # 从 1e-2 降到 3e-3
        weight_decay=1e-4
    )

    best_val_mae = float("inf")
    best_epoch = 0
    patience = 20
    bad_epochs = 0

    os.makedirs("output", exist_ok=True)
    ckpt_head = "output/ensemble_head_ag_mt_best.pt"

    max_epochs = 200
    for ep in range(1, max_epochs + 1):
        head.train()
        train_losses = []

        for x, y, m in train_ld:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            with torch.no_grad():
                pred_a = align_pred_shape(agcrn(x), y)
                pred_m = align_pred_shape(mtgnn(x), y)

            pred = head(pred_a, pred_m)           # [B, H, N]
            loss = masked_mae(pred, y, m)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_mae = float(np.mean(train_losses))

        # ---- 验证集评估 ----
        val_mae, val_rmse = eval_fusion_head(agcrn, mtgnn, head, val_ld, device)

        print(
            f"[ENSEMBLE-HEAD] Epoch {ep:03d} | "
            f"train MAE {train_mae:.4f} | "
            f"val MAE {val_mae:.4f} | val RMSE {val_rmse:.4f}"
        )

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_epoch = ep
            bad_epochs = 0
            torch.save(head.state_dict(), ckpt_head)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"[ENSEMBLE-HEAD] Early stop at epoch {ep}, "
                    f"best epoch {best_epoch}, best val MAE {best_val_mae:.4f}"
                )
                break

    # ===== 用最佳 head 在 TEST 上评估 =====
    head.load_state_dict(torch.load(ckpt_head, map_location=device))
    test_mae, test_rmse = eval_fusion_head(agcrn, mtgnn, head, test_ld, device)
    print(
        f"\n[ENSEMBLE-HEAD TEST] MAE {test_mae:.4f} | RMSE {test_rmse:.4f}"
    )


if __name__ == "__main__":
    main()