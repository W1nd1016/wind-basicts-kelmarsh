# tools/train_s2_agcrn_FnpOnlyAnalysis.py
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset_scada_cerra_s2_FnpOnlyAnalysis import WindDatasetScadaCerraS2
from models.fnp_fusion_OnlyAnalysis_noRot_paper1_plus_v2_wake_downwind import FNPFusionOnlyAnalysis_NoRot_Paper1PlusV2
from models.agcrn_s2_wrapper_FnpOnlyAnalysis import S2Model
from models.agcrn_seq2seq_baseline2_FapOnlyAnalysis import AGCRNSeq2SeqBaseline


def masked_mae(pred, y, m):
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)

def masked_rmse(pred, y, m):
    return torch.sqrt((m * (pred - y) ** 2).sum() / (m.sum() + 1e-6))


@torch.no_grad()
def eval_one(model, loader, device, coords, pos):
    model.eval()
    maes, rmses = [], []
    for x, xv, y, m, fc0, fc0v in loader:
        x, xv, y, m, fc0, fc0v = (
            x.to(device), xv.to(device), y.to(device),
            m.to(device), fc0.to(device), fc0v.to(device)
        )

        x_obs = x[..., :7]
        x_an  = x[..., 7:]
        x_an_v = xv[..., 7:]

        pred = model(
            x_obs=x_obs,
            x_an=x_an,
            coords=coords,
            fc0=fc0,
            pos=pos,
            x_an_valid=x_an_v,
            fc0v=fc0v,
        )
        maes.append(masked_mae(pred, y, m).item())
        rmses.append(masked_rmse(pred, y, m).item())
    return float(np.mean(maes)), float(np.mean(rmses))


def build_coords(meta, L, device):
    xy = np.array(meta["turbine_xy"], dtype=np.float32)  # (N,2)
    N = xy.shape[0]
    t = np.linspace(-1.0, 1.0, L, dtype=np.float32)[:, None]  # (L,1)

    coords = []
    for i in range(L):
        ti = np.repeat(t[i:i+1], N, axis=0)   # (N,1)
        coords.append(np.concatenate([ti, xy], axis=1))  # (N,3)
    coords = np.concatenate(coords, axis=0)  # (P,3)
    return torch.tensor(coords, dtype=torch.float32, device=device)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = "data/wind_scada_cerra_v1_s2_FnpOnlyAnalysis_dataset2"
    meta = json.load(open(f"{root}/meta.json"))

    N = len(meta["turbine_ids"])
    L = int(meta["L"])
    H = int(meta["H"])
    K = int(meta["cerra"]["K"])
    print("N,L,H,K =", N, L, H, K)

    pos_np = np.load(f"{root}/pos.npy").astype(np.float32)   # (N,K,3)
    pos = torch.tensor(pos_np, dtype=torch.float32, device=device)

    coords = build_coords(meta, L=L, device=device)

    train_ds = WindDatasetScadaCerraS2(root=root, split="train", L=L, H=H)
    val_ds   = WindDatasetScadaCerraS2(root=root, split="val",   L=L, H=H)
    test_ds  = WindDatasetScadaCerraS2(root=root, split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    d_model = 128

    # keep total exog dim = 64 (32 scalar + 32 vector)
    fc_emb_scalar = 32
    fc_emb_vec = 32
    exog_dim = fc_emb_scalar + fc_emb_vec

    fusion = FNPFusionOnlyAnalysis_NoRot_Paper1PlusV2(
        d_model=128,
        K_bg=16,
        use_wake=True,                 # (A) 开
        use_downwind_penalty=True,     # (B) 开
        use_angle_calib=False,         # v2-4 默认关
    ).to(device)


    agcrn = AGCRNSeq2SeqBaseline(
        num_nodes=N,
        input_dim=d_model,
        hidden_dim=64,
        embed_dim=10,
        horizon=H,
        K=2,
        topk=None,
        dropout=0.1,
        exog_dim=exog_dim,
    ).to(device)

    model = S2Model(
        fnp_fusion=fusion,
        agcrn_model=agcrn,
        d_model=d_model,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_val = 1e9
    bad = 0
    patience = 30
    os.makedirs("output", exist_ok=True)
    ckpt = "output/s2_fnp_agcrn_vecbranch_best.pt"

    max_epochs = 200
    for ep in range(1, max_epochs + 1):
        model.train()
        losses = []
        tf_ratio = max(0.0, 1.0 - ep / 80.0)

        for x, xv, y, m, fc0, fc0v in train_ld:
            x, xv, y, m, fc0, fc0v = (
                x.to(device), xv.to(device), y.to(device),
                m.to(device), fc0.to(device), fc0v.to(device)
            )

            x_obs = x[..., :7]
            x_an  = x[..., 7:]
            x_an_v = xv[..., 7:]

            pred = model(
                x_obs=x_obs,
                x_an=x_an,
                coords=coords,
                fc0=fc0,
                pos=pos,
                x_an_valid=x_an_v,
                fc0v=fc0v,
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

        train_mae = float(np.mean(losses))
        val_mae, val_rmse = eval_one(model, val_ld, device, coords, pos)
        print(f"[S2][FNP(vec-branch+nac-frame)+AGCRN][E={exog_dim}={fc_emb_scalar}+{fc_emb_vec}] Ep {ep:03d} tf={tf_ratio:.2f} | train {train_mae:.4f} | val MAE {val_mae:.4f} RMSE {val_rmse:.4f}")

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
    test_mae, test_rmse = eval_one(model, test_ld, device, coords, pos)
    print(f"[TEST][S2][FNP(vec-branch)+AGCRN][E={exog_dim}] MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")


if __name__ == "__main__":
    main()
