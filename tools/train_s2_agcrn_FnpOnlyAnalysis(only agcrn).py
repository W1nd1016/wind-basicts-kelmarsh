import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset_scada_cerra_s2_FnpOnlyAnalysis import WindDatasetScadaCerraS2
from models.agcrn_seq2seq_baseline2_FapOnlyAnalysis import AGCRNSeq2SeqBaseline


def masked_mae(pred, y, m):
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)

def masked_rmse(pred, y, m):
    return torch.sqrt((m * (pred - y) ** 2).sum() / (m.sum() + 1e-6))


def build_x_aug(x, fc0):
    """
    x:   (B,L,N,Fx)    where Fx=71 (SCADA+analysis)
    fc0: (B,H,N,K,4)   forecast block
    return x_aug: (B,L,N,Fx + 4H)
    """
    B, L, N, Fx = x.shape
    _, H, N2, K, C = fc0.shape
    assert N2 == N and C == 4

    # mean over K neighbors -> (B,H,N,4)
    fc_mean = fc0.mean(dim=3)

    # flatten (H,4) -> (4H)
    fc_flat = fc_mean.permute(0, 2, 1, 3).reshape(B, N, H * 4)

    # repeat over L -> (B,L,N,4H)
    fc_rep = fc_flat.unsqueeze(1).expand(B, L, N, H * 4)

    # concat -> (B,L,N,Fx+4H)
    return torch.cat([x, fc_rep], dim=-1)


@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    maes, rmses = [], []
    for x, xv, y, m, fc0, fc0v in loader:
        x, y, m, fc0 = x.to(device), y.to(device), m.to(device), fc0.to(device)

        x_aug = build_x_aug(x, fc0)
        pred = model(x_aug)  # (B,H,N)

        maes.append(masked_mae(pred, y, m).item())
        rmses.append(masked_rmse(pred, y, m).item())
    return float(np.mean(maes)), float(np.mean(rmses))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # change root as needed
    root = "data/wind_scada_cerra_v1_s2_FnpOnlyAnalysis_dataset3"
    meta = json.load(open(f"{root}/meta.json"))

    N = len(meta["turbine_ids"])
    L = int(meta["L"])
    H = int(meta["H"])
    K = int(meta["cerra"]["K"])
    print("N,L,H,K =", N, L, H, K)

    train_ds = WindDatasetScadaCerraS2(root=root, split="train", L=L, H=H)
    val_ds   = WindDatasetScadaCerraS2(root=root, split="val",   L=L, H=H)
    test_ds  = WindDatasetScadaCerraS2(root=root, split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    # input_dim = 71 + 4H
    input_dim = 7 + (K * 4) + (H * 4)
    print("input_dim =", input_dim)

    model = AGCRNSeq2SeqBaseline(
        num_nodes=N,
        input_dim=input_dim,
        hidden_dim=64,
        embed_dim=10,
        horizon=H,
        K=2,
        topk=None,
        dropout=0.1,
        exog_dim=0,   # simplest: no separate exog
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_val = 1e9
    bad = 0
    patience = 30
    os.makedirs("output", exist_ok=True)
    ckpt = "output/agcrn_concat_all_best.pt"

    max_epochs = 200
    for ep in range(1, max_epochs + 1):
        model.train()
        losses = []
        tf_ratio = max(0.0, 1.0 - ep / 80.0)

        for x, xv, y, m, fc0, fc0v in train_ld:
            x, y, m, fc0 = x.to(device), y.to(device), m.to(device), fc0.to(device)

            x_aug = build_x_aug(x, fc0)
            pred = model(x_aug, teacher_forcing_y=y, teacher_forcing_ratio=tf_ratio)

            loss = masked_mae(pred, y, m)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        train_mae = float(np.mean(losses))
        val_mae, val_rmse = eval_one(model, val_ld, device)
        print(f"[AGCRN][ConcatAll] Ep {ep:03d} tf={tf_ratio:.2f} | train {train_mae:.4f} | val MAE {val_mae:.4f} RMSE {val_rmse:.4f}")

        if val_mae < best_val - 1e-4:
            best_val = val_mae
            bad = 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("Early stop. best val =", best_val)
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_mae, test_rmse = eval_one(model, test_ld, device)
    print(f"[TEST][AGCRN][ConcatAll] MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")


if __name__ == "__main__":
    main()