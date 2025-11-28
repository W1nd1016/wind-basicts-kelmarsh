import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.wind_dataset import WindSTFDataset
from models.agcrn_min import AGCRN

def masked_mae(pred, y, m):
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

    meta = json.load(open("data/wind/meta.json"))
    input_dim = len(meta["feature_names"])
    print("input_dim =", input_dim)

    train_ds = WindSTFDataset(split="train")
    val_ds   = WindSTFDataset(split="val")
    test_ds  = WindSTFDataset(split="test")

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_ld  = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = AGCRN(
        num_nodes=6,
        input_dim=input_dim,
        hidden_dim=32,
        embed_dim=6,
        horizon=6
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best = 1e9
    patience = 15
    bad = 0

    for ep in range(1, 201):
        model.train()
        train_maes = []
        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            loss = masked_mae(pred, y, m)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_maes.append(loss.item())
        train_mae = float(np.mean(train_maes))

        val_mae, val_rmse = eval_one(model, val_ld, device)
        print(f"Epoch {ep:03d} | train MAE {train_mae:.4f} | val MAE {val_mae:.4f} | val RMSE {val_rmse:.4f}")

        if val_mae < best - 1e-4:
            best = val_mae
            bad = 0
            os.makedirs("output", exist_ok=True)
            torch.save(model.state_dict(), "output/agcrn_min_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {ep}, best val MAE {best:.4f}")
                break

    model.load_state_dict(torch.load("output/agcrn_min_best.pt", map_location=device))
    test_mae, test_rmse = eval_one(model, test_ld, device)
    print(f"[TEST] MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")

if __name__ == "__main__":
    main()