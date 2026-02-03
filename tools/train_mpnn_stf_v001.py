# tools/train_mpnn_stf_v001.py
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.wind_dataset_onlyscada_no_5b import WindSTFDatasetOnlyScada5B
from models.mpnn_stf_v001 import MPNN_STF_v001


def masked_mae(pred, y, m):
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)

def masked_rmse(pred, y, m):
    return torch.sqrt((m * (pred - y) ** 2).sum() / (m.sum() + 1e-6))

@torch.no_grad()
def eval_one_with_3h6h(model, loader, A, device, forward_fn=None):
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
            pred = model(x,A)  # (B,H,N)
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

    root = "data/wind_onlyscada_no_5b_dataset2"

    # ---- meta ----
    with open(f"{root}/meta.json", "r") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    input_dim = len(feature_names)
    turbine_ids = meta.get("turbine_ids", None)
    num_nodes = len(turbine_ids) if turbine_ids is not None else int(meta.get("num_nodes", 6))
    L = int(meta.get("L", 9))
    H = int(meta.get("H", 6))

    print("num_nodes =", num_nodes)
    print("input_dim =", input_dim, "features =", feature_names)
    print("L =", L, "H =", H)

    # ---- adjacency ----
    adj_path = f"{root}/adj.npy"
    adj = np.load(adj_path).astype(np.float32)  # (N,N)
    A = torch.tensor(adj, dtype=torch.float32, device=device)

    # ---- dataset ----
    train_ds = WindSTFDatasetOnlyScada5B(root=root, split="train", L=L, H=H)
    val_ds   = WindSTFDatasetOnlyScada5B(root=root, split="val",   L=L, H=H)
    test_ds  = WindSTFDatasetOnlyScada5B(root=root, split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    print("len(train)=", len(train_ds), "len(val)=", len(val_ds), "len(test)=", len(test_ds))

    # ---- model ----
    model = MPNN_STF_v001(
        num_nodes=num_nodes,
        input_dim=input_dim,   # SCADA-only is 7 in your preprocessing
        horizon=H,
        hidden_dim=96,         # 稍微大一点，baseline 更“论文级”
        gru_layers=1,
        mp_steps=2,
        dropout=0.2,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MPNN_STF_v001 parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_val = 1e9
    patience = 25
    bad = 0

    os.makedirs("output", exist_ok=True)
    ckpt_path = "output/mpnn_stf_v001_best.pt"

    max_epochs = 150
    for ep in range(1, max_epochs + 1):
        model.train()
        losses = []

        for x, y, m in train_ld:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            pred = model(x, A)
            loss = masked_mae(pred, y, m)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())

        scheduler.step()

        train_mae = float(np.mean(losses))
        val_mae3, val_rmse3, val_mae6, val_rmse6 = eval_one_with_3h6h(model, val_ld,A, device)
        print(f"[mpnn-official] Ep {ep:03d} | train {train_mae:.4f} | val MAE 3h {val_mae3:.4f} RMSE 3h {val_rmse3:.4f} val MAE 6h {val_mae6:.4f} RMSE 6h {val_rmse6:.4f}")


        if val_mae6 < best_val - 1e-4:
            best_val = val_mae6
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {ep}, best val MAE {best_val:.4f}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_mae3, test_rmse3, test_mae6, test_rmse6= eval_one_with_3h6h(model, test_ld,A, device)
    print(f"[TEST][MPNN-oficial] MAE 3h {test_mae3:.4f}  RMSE 3h {test_rmse3:.4f} | MAE 6h {test_mae6:.4f} | RMSE 6h {test_rmse6:.4f}")


if __name__ == "__main__":
    main()
