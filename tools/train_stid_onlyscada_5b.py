import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.wind_dataset3 import WindSTFDatasetOnlyScada5B
from models.stid_min import STID


def masked_mae(pred, y, m):
    # pred,y,m: (B,H,N)
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)


def eval_one(model, loader, device):
    model.eval()
    maes, rmses = [], []
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            mae = masked_mae(pred, y, m).item()
            rmse = torch.sqrt((m * (pred - y) ** 2).sum() / (m.sum() + 1e-6)).item()
            maes.append(mae)
            rmses.append(rmse)
    return float(np.mean(maes)), float(np.mean(rmses))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ✅ only-scada 数据目录
    root = "data/wind_onlyscada"
    meta_path = f"{root}/meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    input_dim = len(meta["feature_names"])
    L = int(meta.get("L", 9))
    H = int(meta.get("H", 6))

    print("root =", root)
    print("input_dim =", input_dim)
    print("L =", L, "H =", H)
    print("features =", meta["feature_names"])

    # ✅ Dataset / Loader
    train_ds = WindSTFDatasetOnlyScada5B(root=root, split="train", L=L, H=H)
    val_ds   = WindSTFDatasetOnlyScada5B(root=root, split="val",   L=L, H=H)
    test_ds  = WindSTFDatasetOnlyScada5B(root=root, split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=64, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # ✅ STID
    model = STID(
        num_nodes=6,
        input_dim=input_dim,
        d_model=48,
        hidden_dim=96,
        horizon=H,
        dropout=0.2,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    best = 1e9
    patience = 40
    bad = 0

    for ep in range(1, 301):
        model.train()
        train_losses = []

        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)  # x:(B,L,N,F) y/m:(B,H,N)
            pred = model(x)                                     # (B,H,N)
            loss = masked_mae(pred, y, m)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_losses.append(loss.item())

        train_mae = float(np.mean(train_losses))
        val_mae, val_rmse = eval_one(model, val_ld, device)

        print(f"Epoch {ep:03d} | train MAE {train_mae:.4f} | val MAE {val_mae:.4f} | val RMSE {val_rmse:.4f}")

        if val_mae < best - 1e-4:
            best = val_mae
            bad = 0
            os.makedirs("output", exist_ok=True)
            torch.save(model.state_dict(), "output/stid_onlyscada_5b_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {ep}, best val MAE {best:.4f}")
                break

    model.load_state_dict(torch.load("output/stid_onlyscada_5b_best.pt", map_location=device))
    test_mae, test_rmse = eval_one(model, test_ld, device)
    print(f"[TEST] STID only-scada-5B  MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")


if __name__ == "__main__":
    main()
