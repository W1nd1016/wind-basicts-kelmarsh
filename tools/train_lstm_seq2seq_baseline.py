import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset_onlyscada_no_5b import WindSTFDatasetOnlyScada5B
from models.lstm_seq2seq_baseline import LSTMSeq2SeqBaseline


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
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    meta = json.load(open("data/wind_onlyscada_no_5b_dataset2/meta.json"))
    feature_names = meta["feature_names"]
    F = len(feature_names)
    N = len(meta.get("turbine_ids"))
    L = int(meta.get("L", 9))
    H = int(meta.get("H", 6))
    print("N,F,L,H =", N, F, L, H)
    print("features =", feature_names)

    train_ds = WindSTFDatasetOnlyScada5B(root="data/wind_onlyscada_no_5b_dataset2", split="train", L=L, H=H)
    val_ds   = WindSTFDatasetOnlyScada5B(root="data/wind_onlyscada_no_5b_dataset2", split="val",   L=L, H=H)
    test_ds  = WindSTFDatasetOnlyScada5B(root="data/wind_onlyscada_no_5b_dataset2", split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    model = LSTMSeq2SeqBaseline(
        num_nodes=N,
        input_dim=F,
        hidden_dim=256,      # baseline 要“像样”，别太小
        num_layers=2,
        horizon=H,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_val = 1e9
    bad = 0
    patience = 30
    os.makedirs("output", exist_ok=True)
    ckpt = "output/lstm_seq2seq_baseline_best.pt"

    max_epochs = 200
    for ep in range(1, max_epochs + 1):
        model.train()
        losses = []

        # scheduled sampling：前期强 teacher forcing，后期逐渐减少
        tf_ratio = max(0.0, 1.0 - ep / 80.0)  # 1 -> 0

        for x, y, m in train_ld:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x, teacher_forcing_y=y, teacher_forcing_ratio=tf_ratio)
            loss = masked_mae(pred, y, m)  # 直接用 MAE，更对齐指标

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        train_mae = float(np.mean(losses))
        val_mae3, val_rmse3, val_mae6, val_rmse6 = eval_one_with_3h6h(model, val_ld, device)
        print(f"[LSTM-Seq2Seq] Ep {ep:03d} tf={tf_ratio:.2f} | train {train_mae:.4f} | val MAE 3h {val_mae3:.4f} RMSE 3h {val_rmse3:.4f} | val MAE 6h {val_mae6:.4f} RMSE 6h {val_rmse6:.4f}")

        if val_mae6 < best_val - 1e-4:
            best_val = val_mae6
            bad = 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("Early stop. best val =", best_val)
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_mae3, test_rmse3, test_mae6, test_rmse6= eval_one_with_3h6h(model, test_ld, device)
    print(f"[TEST][LSTM-Seq2Seq] MAE 3h {test_mae3:.4f} | RMSE 3h {test_rmse3:.4f} | MAE 6h {test_mae6:.4f} | RMSE 6h {test_rmse6:.4f}")


if __name__ == "__main__":
    main()
