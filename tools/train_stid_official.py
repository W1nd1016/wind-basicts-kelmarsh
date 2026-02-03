# tools/train_stid_official.py
import os
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

# 加到路径里，方便 import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset_onlyscada_no_5b import WindSTFDatasetOnlyScada5B
from models.stid_official import STID


def masked_mae(pred, y, m):
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)

def masked_rmse(pred, y, m):
    return torch.sqrt((m * (pred - y) ** 2).sum() / (m.sum() + 1e-6))

@torch.no_grad()
def eval_one_with_3h6h(model, loader, device, forward_fn=None):
    """
    返回：MAE<3, RMSE<3, MAE<6, RMSE<6
    forward_fn: 可选，用于像 FNP+AGCRN 那种 forward 需要额外输入时自定义 forward
    """
    model.eval()
    mae3s, rmse3s, mae6s, rmse6s = [], [], [], []

    for batch in loader:
        if forward_fn is None:
            # only-scada baseline: batch=(x,y,m)
            x, y, m = batch
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)  # (B,H,N)
        else:
            # 复杂模型：让你自己提供 forward_fn(batch)->pred,y,m
            pred, y, m = forward_fn(batch, device)

        # <3h: 前3步
        mae3s.append(masked_mae(pred[:, :3], y[:, :3], m[:, :3]).item())
        rmse3s.append(masked_rmse(pred[:, :3], y[:, :3], m[:, :3]).item())

        # <6h: 全部6步
        mae6s.append(masked_mae(pred, y, m).item())
        rmse6s.append(masked_rmse(pred, y, m).item())

    return (
        float(np.mean(mae3s)), float(np.mean(rmse3s)),
        float(np.mean(mae6s)), float(np.mean(rmse6s)),
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 读取 meta.json 看一下特征名 / 维度
    meta_path = os.path.join("data", "wind_onlyscada_no_5b_dataset2", "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names", None)
    input_dim = len(feature_names) if feature_names is not None else meta["input_dim"]
    print("input_dim =", input_dim)
    if feature_names is not None:
        print("features =", feature_names)

    # ===== 数据集 & DataLoader =====
    train_ds = WindSTFDatasetOnlyScada5B(split="train")  # 默认 L=24, H=6
    val_ds = WindSTFDatasetOnlyScada5B(split="val")
    test_ds = WindSTFDatasetOnlyScada5B(split="test")

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
        val_mae3, val_rmse3, val_mae6, val_rmse6 = eval_one_with_3h6h(model, val_ld, device)
        print(f"[stid-official] Ep {ep:03d} | train {train_mae:.4f} | val MAE 3h {val_mae3:.4f} RMSE 3h {val_rmse3:.4f} val MAE 6h {val_mae6:.4f} RMSE 6h {val_rmse6:.4f}")


        # 早停 & 保存
        if val_mae6 < best - 1e-4:
            best = val_mae6
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
    test_mae3, test_rmse3, test_mae6, test_rmse6= eval_one_with_3h6h(model, test_ld, device)
    print(f"[TEST][stid-official] MAE 3h {test_mae3:.4f}  RMSE 3h {test_rmse3:.4f} | MAE 6h {test_mae6:.4f} | RMSE 6h {test_rmse6:.4f}")


if __name__ == "__main__":
    main()