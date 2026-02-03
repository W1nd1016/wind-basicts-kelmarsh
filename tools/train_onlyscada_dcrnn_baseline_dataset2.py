# tools/train_onlyscada_dcrnn_baseline_dataset2.py
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset_onlyscada_no_5b import WindSTFDatasetOnlyScada5B
from models.dcrnn_seq2seq_baseline import DCRNNSeq2Seq


def masked_mae(pred, y, m):
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)

def masked_rmse(pred, y, m):
    return torch.sqrt((m * (pred - y) ** 2).sum() / (m.sum() + 1e-6))


@torch.no_grad()
def eval_one(model, loader, device, supports):
    model.eval()
    mae3s, rmse3s, mae6s, rmse6s = [], [], [], []
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        pred = model(x, supports=supports)  # no teacher forcing

        mae3s.append(masked_mae(pred[:, :3], y[:, :3], m[:, :3]).item())
        rmse3s.append(masked_rmse(pred[:, :3], y[:, :3], m[:, :3]).item())

        mae6s.append(masked_mae(pred, y, m).item())
        rmse6s.append(masked_rmse(pred, y, m).item())

    return (
        float(np.mean(mae3s)), float(np.mean(rmse3s)),
        float(np.mean(mae6s)), float(np.mean(rmse6s)),
    )


def build_supports_from_adj(adj_np: np.ndarray, device):
    """
    DCRNN 常用两种 support:
      A_rw      : row-normalized adjacency (random walk)
      A_rw_rev  : row-normalized transpose adjacency (reverse random walk)
    你的 adj.npy 本身就是行归一化过的（构建脚本里 A = A / row_sum）
    这里仍然稳妥地再做一次归一化，并构造 transpose 的归一化版本。
    """
    A = adj_np.astype(np.float32)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-6)

    AT = A.T
    AT = AT / (AT.sum(axis=1, keepdims=True) + 1e-6)

    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    AT_t = torch.tensor(AT, dtype=torch.float32, device=device)
    return [A_t, AT_t]


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = "data/wind_onlyscada_no_5b_dataset2"  # 你要跑哪个数据集就改这里
    meta = json.load(open(f"{root}/meta.json"))

    F = len(meta["feature_names"])
    N = len(meta.get("turbine_ids", []))
    L = int(meta.get("L", 9))
    H = int(meta.get("H", 6))
    print("N,F,L,H =", N, F, L, H)
    print("features =", meta["feature_names"])

    # ---- supports from adj.npy ----
    adj_np = np.load(f"{root}/adj.npy").astype(np.float32)  # (N,N)
    supports = build_supports_from_adj(adj_np, device=device)

    train_ds = WindSTFDatasetOnlyScada5B(root=root, split="train", L=L, H=H)
    val_ds   = WindSTFDatasetOnlyScada5B(root=root, split="val",   L=L, H=H)
    test_ds  = WindSTFDatasetOnlyScada5B(root=root, split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    model = DCRNNSeq2Seq(
        num_nodes=N,
        input_dim=F,     # only-scada: 7
        hidden_dim=64,
        horizon=H,
        K=2,             # diffusion steps
        num_supports=len(supports),
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_val = 1e9
    bad = 0
    patience = 30
    os.makedirs("output", exist_ok=True)
    ckpt = "output/dcrnn_seq2seq_onlyscada_best.pt"

    max_epochs = 200
    for ep in range(1, max_epochs + 1):
        model.train()
        losses = []

        tf_ratio = max(0.0, 1.0 - ep / 80.0)  # scheduled sampling

        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)

            pred = model(
                x,
                supports=supports,
                teacher_forcing_y=y,
                teacher_forcing_ratio=tf_ratio,
            )
            loss = masked_mae(pred, y, m)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        train_mae6 = float(np.mean(losses))

        val_mae3, val_rmse3, val_mae6, val_rmse6 = eval_one(model, val_ld, device, supports)
        print(
            f"[DCRNN-Seq2Seq] Ep {ep:03d} tf={tf_ratio:.2f} | "
            f"train <6h MAE {train_mae6:.4f} | "
            f"val <3h MAE {val_mae3:.4f} RMSE {val_rmse3:.4f} | "
            f"val <6h MAE {val_mae6:.4f} RMSE {val_rmse6:.4f}"
        )

        # 用 <6h MAE 做 early stop / best
        if val_mae6 < best_val - 1e-4:
            best_val = val_mae6
            bad = 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("Early stop. best val <6h MAE =", best_val)
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_mae3, test_rmse3, test_mae6, test_rmse6 = eval_one(model, test_ld, device, supports)
    print(
        f"[TEST][DCRNN-Seq2Seq] "
        f"<3h MAE {test_mae3:.4f} RMSE {test_rmse3:.4f} | "
        f"<6h MAE {test_mae6:.4f} RMSE {test_rmse6:.4f}"
    )


if __name__ == "__main__":
    main()
