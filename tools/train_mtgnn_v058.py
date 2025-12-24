# tools/train_mtgnn_v058.py
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.wind_dataset import WindSTFDataset
from models.mtgnn_v058 import MTGNN_v058


def masked_mae(pred, y, m):
    """
    pred, y, m: [B, H, N]
    m: mask (1 表示有效)
    """
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)


def eval_one(model, loader, device):
    model.eval()
    maes, rmses = [], []
    with torch.no_grad():
        for x, y, m in loader:
            # x: [B, L, N, F], y/m: [B, H, N]
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            pred = model(x)  # [B, H, N]

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

    # ===== 读取 meta.json，拿到特征维度 & 节点数 =====
    meta_path = "data/wind/meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    input_dim = len(feature_names)
    print("input_dim =", input_dim)
    print("features =", feature_names)

    # 节点数：用涡轮机 ID 的数量
    turbine_ids = meta.get("turbine_ids", None)
    if turbine_ids is not None:
        num_nodes = len(turbine_ids)
    else:
        num_nodes = meta.get("num_nodes", 6)
    print("num_nodes =", num_nodes)

    # 时序长度 & 预测步长
    L = meta.get("L", meta.get("history_len", 24))
    H = meta.get("H", meta.get("horizon", 6))
    print("L =", L, "H =", H)

    # ===== 加载静态邻接矩阵 =====
    try:
        adj = np.load("data/wind/adj.npy").astype(np.float32)
        print("Loaded adj.npy, shape:", adj.shape)
    except FileNotFoundError:
        print("WARNING: data/wind/adj.npy not found, use identity graph instead.")
        adj = np.eye(num_nodes, dtype=np.float32)

    # ===== 数据集 & DataLoader =====
    train_ds = WindSTFDataset(split="train", L=L, H=H)
    val_ds   = WindSTFDataset(split="val",   L=L, H=H)
    test_ds  = WindSTFDataset(split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    # ===== 构建 MTGNN_v058 模型 =====
    model = MTGNN_v058(
        num_nodes=num_nodes,
        in_dim=input_dim,   # 55 维（P, dP, W, dir_sin/cos, nac_sin/cos + CERRA 全部）
        seq_length=L,       # 24
        horizon=H,          # 6
        predefined_A=adj,
        gcn_true=True,
        buildA_true=False,  # 使用静态 adj，不学习图
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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MTGNN_v058 parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-4)

    # 在 opt 定义下面加一个 scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,   # 每 5 个 epoch 降一次
        gamma=0.7      # 乘 0.7
    )
    best_val_mae = 1e9
    patience = 20
    bad_epochs = 0

    # ===== 训练循环 =====
    max_epochs = 150
    for ep in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for x, y, m in train_ld:
            x = x.to(device)   # [B, L, N, F]
            y = y.to(device)   # [B, H, N]
            m = m.to(device)   # [B, H, N]

            pred = model(x)    # [B, H, N]
            loss = masked_mae(pred, y, m)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_mae = float(np.mean(train_losses))
        val_mae, val_rmse = eval_one(model, val_ld, device)

        print(f"[VAL] MAE {val_mae:.4f} | [VAL] RMSE {val_rmse:.4f}")
        print(
            f"[MTGNN_v058] Epoch {ep:03d} | "
            f"train MAE {train_mae:.4f} | val MAE {val_mae:.4f} | val RMSE {val_rmse:.4f}"
        )
        scheduler.step()

        # 早停 & 保存最佳模型
        os.makedirs("output", exist_ok=True)
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            bad_epochs = 0
            torch.save(model.state_dict(), "output/mtgnn_v058_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"[MTGNN_v058] Early stop at epoch {ep}, "
                    f"best val MAE {best_val_mae:.4f}"
                )
                break

    # ===== 测试集评估 =====
    model.load_state_dict(torch.load("output/mtgnn_v058_best.pt", map_location=device))
    test_mae, test_rmse = eval_one(model, test_ld, device)
    print(f"[MTGNN_v058 TEST] MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")


if __name__ == "__main__":
    main()