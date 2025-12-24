# tools/train_stid_official.py
import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

# 加到路径里，方便 import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset import WindSTFDataset
from models.stid_official import STID


def masked_mae(pred, y, m):
    # pred, y: [B, H, N]
    # m:       [B, H, N]
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)


def eval_one(model, loader, device):
    model.eval()
    maes, rmses = [], []
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
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

    # 读取 meta.json 看一下特征名 / 维度
    meta_path = os.path.join("data", "wind", "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names", None)
    input_dim = len(feature_names) if feature_names is not None else meta["input_dim"]
    print("input_dim =", input_dim)
    if feature_names is not None:
        print("features =", feature_names)

    # ===== 数据集 & DataLoader =====
    train_ds = WindSTFDataset(split="train")  # 默认 L=24, H=6
    val_ds = WindSTFDataset(split="val")
    test_ds = WindSTFDataset(split="test")

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=64, shuffle=False)

    # ===== 官方 STID 架构 + 适配 wrapper =====
    model = STID(
        num_nodes=6,
        input_dim=input_dim,
        input_len=train_ds.L,  # 通常 24
        horizon=train_ds.H,    # 通常 6
        node_dim=16,
        embed_dim=64,
        num_layer=3,
        if_node=True,
        if_T_i_D=False,  # 不用额外时间特征
        if_D_i_W=False,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best = 1e9
    patience = 20
    bad = 0

    for ep in range(1, 301):
        model.train()
        train_losses = []
        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            loss = masked_mae(pred, y, m)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_losses.append(loss.item())

        train_mae = float(np.mean(train_losses))
        val_mae, val_rmse = eval_one(model, val_ld, device)

        print(
            f"Epoch {ep:03d} | train MAE {train_mae:.4f} | "
            f"val MAE {val_mae:.4f} | val RMSE {val_rmse:.4f}"
        )

        # 早停 & 保存
        if val_mae < best - 1e-4:
            best = val_mae
            bad = 0
            os.makedirs("output", exist_ok=True)
            torch.save(model.state_dict(), "output/stid_official_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {ep}, best val MAE {best:.4f}")
                break

    # ===== 用最优模型在 test 上评估 =====
    model.load_state_dict(
        torch.load("output/stid_official_best.pt", map_location=device)
    )
    test_mae, test_rmse = eval_one(model, test_ld, device)
    print(f"[TEST] STID-official MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")


if __name__ == "__main__":
    main()