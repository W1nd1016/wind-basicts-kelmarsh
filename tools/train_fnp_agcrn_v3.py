# tools/train_fnp_agcrn_v3.py
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset_scada_cerra_fnp_v2 import WindDatasetScadaCerraFNPv2
from models.fnp_fusion_v3 import FNPFusionV3
from models.agcrn_seq2seq_baseline import AGCRNSeq2SeqBaseline

def masked_mae(pred, y, m):
    return (m * (pred - y).abs()).sum() / (m.sum() + 1e-6)

def masked_rmse(pred, y, m):
    return torch.sqrt((m * (pred - y) ** 2).sum() / (m.sum() + 1e-6))

class FNP_AGCRN_ModelV3(torch.nn.Module):
    """
    z_hist(FNP) -> AGCRN -> base
    z_bg_future(FNP) -> correction -> final = base + softplus(gamma)*delta
    """
    def __init__(self, fnp_fusion, agcrn_model, d_model: int):
        super().__init__()
        self.fnp = fnp_fusion
        self.agcrn = agcrn_model
        self.corr = torch.nn.Sequential(
            torch.nn.Linear(d_model, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.gamma = torch.nn.Parameter(torch.tensor(-2.0))  # start small after softplus

    def forward(
        self,
        x_obs, x_bg, coords, dx, dy, ds,
        x_obs_v=None, x_bg_v=None,
        teacher_forcing_y=None,
        teacher_forcing_ratio: float = 0.0,
    ):
        z_hist, z_bg_future = self.fnp(
            x_obs, x_bg, coords, dx, dy, ds,
            x_obs_v=x_obs_v, x_bg_v=x_bg_v
        )
        if teacher_forcing_y is None:
            base = self.agcrn(z_hist)
        else:
            base = self.agcrn(z_hist, teacher_forcing_y=teacher_forcing_y, teacher_forcing_ratio=teacher_forcing_ratio)

        delta = self.corr(z_bg_future).squeeze(-1)  # (B,H,N)
        return base + torch.nn.functional.softplus(self.gamma) * delta

@torch.no_grad()
def eval_one(model, loader, device, coords, dx, dy, ds):
    model.eval()
    maes, rmses = [], []
    for x, xv, y, m in loader:
        x, xv, y, m = x.to(device), xv.to(device), y.to(device), m.to(device)

        x_obs   = x[..., :7]
        x_bg    = x[..., 7:]
        x_obs_v = xv[..., :7]
        x_bg_v  = xv[..., 7:]

        pred = model(x_obs, x_bg, coords, dx, dy, ds, x_obs_v=x_obs_v, x_bg_v=x_bg_v)
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

def _infer_int(x, default=None):
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        try:
            return int(x)
        except Exception:
            return default
    if isinstance(x, (list, tuple)):
        return len(x)
    if isinstance(x, dict):
        return default
    return default

def _load_neighbors_arrays(root: str):
    """
    Try multiple filenames to avoid being tied to one builder naming.
    Returns dx_m, dy_m, dist_m with shape (N,K).
    """
    candidates = [
        ("neighbors_dx_m.npy", "neighbors_dy_m.npy", "neighbors_dist_m.npy"),
        ("neighbors_dx.npy",   "neighbors_dy.npy",   "neighbors_dist.npy"),
        ("neighbors_dx_meters.npy", "neighbors_dy_meters.npy", "neighbors_dist_meters.npy"),
    ]
    for a, b, c in candidates:
        pa = os.path.join(root, a)
        pb = os.path.join(root, b)
        pc = os.path.join(root, c)
        if os.path.exists(pa) and os.path.exists(pb) and os.path.exists(pc):
            dx_m = np.load(pa).astype(np.float32)
            dy_m = np.load(pb).astype(np.float32)
            ds_m = np.load(pc).astype(np.float32)
            return dx_m, dy_m, ds_m

    # also try npz
    npz = os.path.join(root, "neighbors.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        for kset in [("dx_m","dy_m","dist_m"), ("dx","dy","dist")]:
            if all(k in z for k in kset):
                return z[kset[0]].astype(np.float32), z[kset[1]].astype(np.float32), z[kset[2]].astype(np.float32)

    raise FileNotFoundError(
        "Cannot find neighbors arrays in root. Tried:\n"
        + "\n".join([f"  - {x[0]}, {x[1]}, {x[2]}" for x in candidates])
        + "\n  - neighbors.npz (keys dx_m/dy_m/dist_m or dx/dy/dist)"
    )

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = "data/wind_scada_cerra_fnp_v2"
    meta = json.load(open(f"{root}/meta.json"))

    L = int(meta["L"])
    H = int(meta["H"])
    N = len(meta["turbine_ids"])

    bg_maps = meta.get("bg_maps", {})

    # ---- infer K ----
    K = _infer_int(bg_maps.get("k_neighbors", None), default=None)
    if K is None:
        K = 16  # fallback
    # ---- infer bg_dim from meta feature names if available ----
    bg_dim = None
    if "feature_names" in meta and isinstance(meta["feature_names"], list):
        bg_dim = len(meta["feature_names"]) - 7

    # ---- infer blocks ----
    blocks = _infer_int(bg_maps.get("blocks", None), default=None)
    if blocks is None:
        blocks = 1 + H  # an + fc1..H

    # ---- infer vars_per_point ----
    vars_per_point = _infer_int(bg_maps.get("vars_per_point", None), default=None)
    if vars_per_point is None and bg_dim is not None:
        denom = blocks * K
        if denom > 0 and (bg_dim % denom == 0):
            vars_per_point = bg_dim // denom
    if vars_per_point is None:
        vars_per_point = 2  # your current v2 seems uv-only

    # ---- infer an_dim safely (THIS fixes your crash) ----
    an_idx = bg_maps.get("an_idx", None)
    if isinstance(an_idx, (int, float, np.integer)):
        an_dim = int(an_idx)
    elif isinstance(an_idx, (list, tuple)):
        an_dim = len(an_idx)
    else:
        # fallback: K * vars_per_point
        an_dim = int(K * vars_per_point)

    print(f"N,L,H,K = {N} {L} {H} {K}")
    print(f"bg inferred: blocks={blocks}, vars_per_point={vars_per_point}, an_dim={an_dim}, bg_dim(meta)={bg_dim}")

    coords = build_coords(meta, L=L, device=device)

    # neighbors in meters -> normalize
    dx_m, dy_m, dist_m = _load_neighbors_arrays(root)  # (N,K)
    dx_scale = float(np.max(np.abs(dx_m)) + 1e-6)
    dy_scale = float(np.max(np.abs(dy_m)) + 1e-6)
    ds_scale = float(np.max(dist_m) + 1e-6)

    dx = (dx_m / dx_scale)[None, None, :, :, None]  # (1,1,N,K,1)
    dy = (dy_m / dy_scale)[None, None, :, :, None]
    ds = (dist_m / ds_scale)[None, None, :, :, None]

    dx = torch.tensor(dx, dtype=torch.float32, device=device)
    dy = torch.tensor(dy, dtype=torch.float32, device=device)
    ds = torch.tensor(ds, dtype=torch.float32, device=device)

    # dataset
    train_ds = WindDatasetScadaCerraFNPv2(root=root, split="train", L=L, H=H)
    val_ds   = WindDatasetScadaCerraFNPv2(root=root, split="val",   L=L, H=H)
    test_ds  = WindDatasetScadaCerraFNPv2(root=root, split="test",  L=L, H=H)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    # model
    d_model = 128
    fnp = FNPFusionV3(
        d_model=d_model,
        modes=8,
        nfl_layers=2,
        K=K,
        H=H,
        vars_per_point=vars_per_point,
    ).to(device)

    agcrn = AGCRNSeq2SeqBaseline(
        num_nodes=N,
        input_dim=d_model,
        hidden_dim=64,
        embed_dim=10,
        horizon=H,
        K=2,
        dropout=0.1,
    ).to(device)

    model = FNP_AGCRN_ModelV3(fnp_fusion=fnp, agcrn_model=agcrn, d_model=d_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_val = 1e9
    bad = 0
    patience = 30
    os.makedirs("output", exist_ok=True)
    ckpt = "output/fnp_agcrn_v3_best.pt"

    max_epochs = 200
    for ep in range(1, max_epochs + 1):
        model.train()
        losses = []
        tf_ratio = max(0.0, 1.0 - ep / 80.0)

        for x, xv, y, m in train_ld:
            x, xv, y, m = x.to(device), xv.to(device), y.to(device), m.to(device)

            x_obs   = x[..., :7]
            x_bg    = x[..., 7:]
            x_obs_v = xv[..., :7]
            x_bg_v  = xv[..., 7:]

            pred = model(
                x_obs, x_bg, coords, dx, dy, ds,
                x_obs_v=x_obs_v, x_bg_v=x_bg_v,
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
        val_mae, val_rmse = eval_one(model, val_ld, device, coords, dx, dy, ds)
        print(f"[FNPv3+AGCRN] Ep {ep:03d} tf={tf_ratio:.2f} | train {train_mae:.4f} | val MAE {val_mae:.4f} RMSE {val_rmse:.4f}")

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
    test_mae, test_rmse = eval_one(model, test_ld, device, coords, dx, dy, ds)
    print(f"[TEST][FNPv3+AGCRN] MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")

if __name__ == "__main__":
    main()
