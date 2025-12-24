# tools/train_lstm_stf_v001.py
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.wind_dataset import WindSTFDataset
from models.lstm_stf_v001 import LSTM_STF_v001


def masked_smooth_l1(pred, y, m, beta=1.0):
    """
    pred, y, m: [B, H, N]
    m: 1 for valid entries, 0 for missing
    """
    loss = F.smooth_l1_loss(pred, y, reduction="none", beta=beta)
    loss = loss * m
    return loss.sum() / (m.sum() + 1e-6)


def masked_mae(pred, y, m):
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)


@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    maes, rmses = [], []
    for x, y, m in loader:
        x = x.to(device)  # [B, L, N, F_full]
        y = y.to(device)  # [B, H, N]
        m = m.to(device)  # [B, H, N]

        pred = model(x)   # [B, H, N]

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
    input_dim = len(feature_names)   # 现在应该是 55（含 CERRA）
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

    # ===== 数据集 & DataLoader =====
    train_ds = WindSTFDataset(split="train", L=L, H=H)
    val_ds   = WindSTFDataset(split="val",   L=L, H=H)
    test_ds  = WindSTFDataset(split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    # ===== 构建 LSTM（仅用 SCADA 特征） =====
    model = LSTM_STF_v001(
        num_nodes=num_nodes,
        full_input_dim=input_dim,  # 例如 55
        horizon=H,
        scada_dim=7,               # 只用前 7 维 SCADA
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LSTM_STF_v001 parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5
    )

    best_val_mae = 1e9
    best_epoch = -1
    patience = 30
    bad_epochs = 0

    os.makedirs("output", exist_ok=True)
    ckpt_path = "output/lstm_stf_v001_best.pt"

    max_epochs = 150
    for ep in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for x, y, m in train_ld:
            x = x.to(device)  # [B, L, N, F_full]
            y = y.to(device)  # [B, H, N]
            m = m.to(device)  # [B, H, N]

            pred = model(x)   # [B, H, N]
            loss = masked_smooth_l1(pred, y, m, beta=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        train_mae = float(np.mean(train_losses))
        val_mae, val_rmse = eval_one(model, val_ld, device)

        print(
            f"[LSTM_STF_v001] Epoch {ep:03d} | "
            f"train MAE {train_mae:.4f} | "
            f"val MAE {val_mae:.4f} | "
            f"val RMSE {val_rmse:.4f}"
        )

        # Early stopping
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_epoch = ep
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"[LSTM_STF_v001] Early stop at epoch {ep}, "
                    f"best epoch {best_epoch}, best val MAE {best_val_mae:.4f}"
                )
                break

    # ===== 测试集评估 =====
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_mae, test_rmse = eval_one(model, test_ld, device)
    print(f"[LSTM_STF_v001 TEST] MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")


if __name__ == "__main__":
    main()