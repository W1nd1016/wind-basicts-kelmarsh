# tools/train_dgcrn_v058.py
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.wind_dataset import WindSTFDataset
from models.dgcrn_v058 import DGCRN


def masked_mae(pred, y, m):
    """
    pred, y, m: [B, H, N]
    """
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)


@torch.no_grad()
def eval_one(model, loader, device, L, H, predefined_A):
    model.eval()
    maes, rmses = [], []
    for x, y, m in loader:
        # x: [B, L, N, 55]
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        B, L_, N, F = x.shape

        # ============ 这里直接用全部 55 维特征 ============
        history = x  # [B, L, N, 55]

        # future_data: [B, H, N, 2]
        # 通道 0 = 未来真实功率，通道 1 = 占位（这里先全 0）
        future = torch.zeros(B, H, N, 2, device=device)
        future[..., 0] = y  # [B, H, N]

        out = model(
            history,
            future,
            batch_seen=0,
            epoch=0,
            train=False,
            task_level=H
        )  # [B, L, N, 1]

        # 只取前 H 步预测
        pred = out[:, :H, :, 0]  # [B, H, N]

        mae = masked_mae(pred, y, m).item()
        rmse = torch.sqrt((m * (pred - y) ** 2).sum() / (m.sum() + 1e-6)).item()
        maes.append(mae)
        rmses.append(rmse)
    return float(np.mean(maes)), float(np.mean(rmses))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- 读取 meta，确定特征名和维度 --------
    meta_path = "data/wind/meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    input_dim = len(feature_names)
    num_nodes = meta.get("num_nodes", 6)

    print("input_dim =", input_dim)
    print("features =", feature_names)

    # 序列长度设置
    L = 24  # 输入长度
    H = 6   # 预测步数

    # -------- 读取静态邻接矩阵 adj.npy --------
    try:
        adj = np.load("data/wind/adj.npy").astype(np.float32)
        print("Loaded adj.npy, shape:", adj.shape)
    except FileNotFoundError:
        print("WARNING: data/wind/adj.npy not found, use identity graph instead.")
        adj = np.eye(num_nodes, dtype=np.float32)

    A1 = torch.tensor(adj, device=device)      # [N, N]
    A2 = torch.tensor(adj.T, device=device)    # [N, N]
    predefined_A = [A1, A2]

    # -------- 数据集 & DataLoader --------
    train_ds = WindSTFDataset(root="data/wind", split="train", L=L, H=H)
    val_ds   = WindSTFDataset(root="data/wind", split="val",   L=L, H=H)
    test_ds  = WindSTFDataset(root="data/wind", split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    # -------- 构建 DGCRN 模型：in_dim = 55（全部特征） --------
    model = DGCRN(
        gcn_depth=2,
        num_nodes=num_nodes,
        predefined_A=predefined_A,
        dropout=0.3,
        subgraph_size=num_nodes,
        node_dim=40,
        middle_dim=2,
        seq_length=L,         # 输入长度 24
        in_dim=input_dim,     # ★★ 使用所有 55 维特征 ★★
        list_weight=[0.05, 0.95, 0.95],
        tanhalpha=3,
        cl_decay_steps=4000,
        rnn_size=64,
        hyperGNN_dim=16
    ).to(device)

    # 为了更稳定，可以先关掉 curriculum learning
    model.use_curriculum_learning = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_mae = float("inf")
    patience = 10
    bad_epochs = 0
    batch_seen = 0

    for epoch in range(1, 151):
        model.train()
        train_losses = []

        for x, y, m in train_ld:
            # x: [B, L, N, 55], y: [B, H, N], m: [B, H, N]
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            B, L_, N, F = x.shape
            batch_seen += 1

            # ============ 不再只取 [0,2]，直接用全部特征 ============
            history = x  # [B, L, N, 55]

            # future_data: [B, H, N, 2]
            # 通道 0: 真实未来功率
            # 通道 1: 时间特征占位（这里为 0）
            future = torch.zeros(B, H, N, 2, device=device)
            future[..., 0] = y

            out = model(
                history,
                future,
                batch_seen=batch_seen,
                epoch=epoch,
                train=True,
                task_level=H
            )  # [B, L, N, 1]

            pred = out[:, :H, :, 0]  # [B, H, N]

            loss = masked_mae(pred, y, m)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_mae = float(np.mean(train_losses))

        # -------- 验证 --------
        val_mae, val_rmse = eval_one(model, val_ld, device, L, H, predefined_A)
        print(
            f"[DGCRN_v058] Epoch {epoch:03d} | "
            f"train MAE {train_mae:.4f} | val MAE {val_mae:.4f} | val RMSE {val_rmse:.4f}"
        )

        # -------- 早停 & 模型保存 --------
        os.makedirs("output", exist_ok=True)
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            bad_epochs = 0
            torch.save(model.state_dict(), "output/dgcrn_v058_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[DGCRN_v058] Early stop at epoch {epoch}, best val MAE {best_val_mae:.4f}")
                break

    # -------- 测试集评估 --------
    model.load_state_dict(torch.load("output/dgcrn_v058_best.pt", map_location=device))
    test_mae, test_rmse = eval_one(model, test_ld, device, L, H, predefined_A)
    print(f"[DGCRN_v058 TEST] MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")


if __name__ == "__main__":
    main()