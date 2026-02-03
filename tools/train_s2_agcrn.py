# tools/train_s2_agcrn.py
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset_scada_cerra_s2 import WindDatasetScadaCerraS2
from models.fnp_fusion import FNPFusion
from models.agcrn_s2_wrapper import S2Model
from models.agcrn_seq2seq_baseline2 import AGCRNSeq2SeqBaseline  # 你自己的实现

def masked_mae(pred, y, m):
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)

def masked_rmse(pred, y, m):
    return torch.sqrt((m * (pred - y) ** 2).sum() / (m.sum() + 1e-6))

@torch.no_grad()
def eval_one(model, loader, device, coords):
    model.eval()
    maes, rmses = [], []
    for x, xv, y, m, fc0, fc0v in loader:
        x, xv, y, m, fc0, fc0v = x.to(device), xv.to(device), y.to(device), m.to(device), fc0.to(device), fc0v.to(device)

        x_obs = x[..., :7]
        x_bg  = x[..., 7:]
        x_bg_v = xv[..., 7:]

        pred = model(x_obs, x_bg, coords, fc0, x_bg_valid=x_bg_v, fc0v=fc0v)
        maes.append(masked_mae(pred, y, m).item())
        rmses.append(masked_rmse(pred, y, m).item())
    return float(np.mean(maes)), float(np.mean(rmses))

def build_coords(meta, L, device):
    """
    coords: (P,3) where P=L*N
      t in [-1,1]
      (x,y) from meta["turbine_xy"]
    """
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

    root = "data/wind_scada_cerra_v1_s2"
    meta = json.load(open(f"{root}/meta.json"))
    N = len(meta["turbine_ids"])
    L = int(meta["L"])
    H = int(meta["H"])

    fc_map = meta["s2"]["cerra_fc_idx_by_lead"]
    f_fc = len(fc_map["1"]) if "1" in fc_map else 0
    print("N,L,H,f_fc =", N, L, H, f_fc)

    coords = build_coords(meta, L=L, device=device)

    train_ds = WindDatasetScadaCerraS2(root=root, split="train", L=L, H=H)
    val_ds   = WindDatasetScadaCerraS2(root=root, split="val",   L=L, H=H)
    test_ds  = WindDatasetScadaCerraS2(root=root, split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    # -------- Build FNP fusion (Route A) --------
    obs_dim = 7
    bg_dim  = len(meta["feature_names"]) - 7
    d_model = 128
    scheme = "A"

    # ✅ BG 支路开启 local SetConv（K=16 内部核卷积）
    fnp = FNPFusion(
        obs_dim=obs_dim,
        bg_dim=bg_dim,
        d_model=d_model,
        scheme=scheme,
        modes=8,
        K_bg=int(meta["s2"].get("k_neighbors", 16)),
        H=H,
        bg_use_setconv=True,   # <-- 关键开关：不想用就 False
    ).to(device)

    # -------- AGCRN head --------
    agcrn = AGCRNSeq2SeqBaseline(
        num_nodes=N,
        input_dim=d_model,
        hidden_dim=64,
        embed_dim=10,
        horizon=H,
        K=2,
        topk=None,  # type: ignore
        dropout=0.1,
    ).to(device)

    model = S2Model(fnp_fusion=fnp, agcrn_model=agcrn, d_model=d_model, f_fc=f_fc).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_val = 1e9
    bad = 0
    patience = 30
    os.makedirs("output", exist_ok=True)
    ckpt = "output/s2_fnp_agcrn_best1.pt"

    max_epochs = 200
    for ep in range(1, max_epochs + 1):
        model.train()
        losses = []
        tf_ratio = max(0.0, 1.0 - ep / 80.0)

        for x, xv, y, m, fc0, fc0v in train_ld:
            x, xv, y, m, fc0, fc0v = x.to(device), xv.to(device), y.to(device), m.to(device), fc0.to(device), fc0v.to(device)

            x_obs = x[..., :7]
            x_bg  = x[..., 7:]
            x_bg_v = xv[..., 7:]

            pred = model(
                x_obs, x_bg, coords, fc0,
                x_bg_valid=x_bg_v, fc0v=fc0v,
                teacher_forcing_y=y, teacher_forcing_ratio=tf_ratio
            )
            loss = masked_mae(pred, y, m)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()

        train_mae = float(np.mean(losses))
        val_mae, val_rmse = eval_one(model, val_ld, device, coords)
        print(f"[S2-FNP-{scheme}+AGCRN][bg_setconv=1] Ep {ep:03d} tf={tf_ratio:.2f} | train {train_mae:.4f} | val MAE {val_mae:.4f} RMSE {val_rmse:.4f}")

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
    test_mae, test_rmse = eval_one(model, test_ld, device, coords)
    print(f"[TEST][S2-FNP-{scheme}+AGCRN][bg_setconv=1] MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")

if __name__ == "__main__":
    main()
