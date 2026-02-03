# tools/eval_persistence_onlyscada_5b.py
import os
import json
import argparse
import numpy as np


def load_np(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/wind_onlyscada_no_5b",
                    help="dataset root containing X.npy, X_valid.npy, Y.npy, mask.npy, meta.json")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--rated_kw", type=float, default=2050.0)
    args = ap.parse_args()

    root = args.root
    meta_path = os.path.join(root, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found at: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # ---- load arrays (only need X_valid, Y, mask) ----
    Xv = load_np(os.path.join(root, "X_valid.npy"))  # (T3,N,F)
    Y  = load_np(os.path.join(root, "Y.npy"))        # (T1,N)  normalized
    M  = load_np(os.path.join(root, "mask.npy"))     # (T1,N)

    # ---- meta params ----
    L = int(meta.get("L", 9))
    H = int(meta.get("H", 6))
    ratio = int(meta["ratio_x_to_y"])        # should be 3
    offset = int(meta["offset_hours"])       # usually 0

    h3 = min(3, H)
    h6 = min(6, H)

    x_start = np.datetime64(meta["x_start"])
    x_step  = np.timedelta64(int(meta["x_step_hours"]), "h")

    splits = meta["splits"]
    if args.split == "train":
        x0, x1 = splits["train_x"]
        y0, y1 = splits["train_y"]
        allowed_years = {2016, 2017, 2018}
    elif args.split == "val":
        x0, x1 = splits["val_x"]
        y0, y1 = splits["val_y"]
        allowed_years = {2019}
    else:
        x0, x1 = splits["test_x"]
        y0, y1 = splits["test_y"]
        allowed_years = {2020}

    x0, x1 = int(x0), int(min(x1, Xv.shape[0]))
    y0, y1 = int(y0), int(min(y1, Y.shape[0]))

    # ---- build valid sample index list exactly like WindSTFDatasetOnlyScada5B (as you wrote) ----
    idxs = []
    for x_idx in range(x0, x1):
        if x_idx < (L - 1):
            continue

        y_idx = offset + x_idx * ratio
        if (y_idx + H) >= len(Y):
            continue

        # split leakage guard: the last forecast hour must be within allowed years
        x_time = x_start + x_idx * x_step
        y_end_time = x_time + np.timedelta64(H, "h")
        year = int(str(y_end_time)[:4])
        if year not in allowed_years:
            continue

        # 5B: any invalid in history window -> drop
        win_v = Xv[x_idx - L + 1 : x_idx + 1]  # (L,N,F)
        if not np.all(win_v > 0.5):
            continue

        # mask: future H hours must have at least one valid point
        m = M[y_idx + 1 : y_idx + H + 1]       # (H,N)
        if float(m.sum()) <= 0.0:
            continue

        # also require y range inside split y bounds
        if (y_idx + H) >= y1 or (y_idx + 1) < y0:
            continue

        idxs.append(x_idx)

    if len(idxs) == 0:
        raise RuntimeError(f"No valid samples found for split={args.split}. "
                           f"Check meta splits / files / X_valid / mask.")

    # ---- evaluate persistence baseline (anchor-hour persistence) ----
    abs_sum3 = 0.0; sq_sum3 = 0.0; m_sum3 = 0.0
    abs_sum6 = 0.0; sq_sum6 = 0.0; m_sum6 = 0.0

    for x_idx in idxs:
        y_idx = offset + x_idx * ratio

        y_true = Y[y_idx + 1 : y_idx + H + 1]     # (H,N)
        m = M[y_idx + 1 : y_idx + H + 1]          # (H,N)

        # baseline: repeat the last observed hour at anchor (Y[y_idx]) for all H steps
        y_anchor = Y[y_idx : y_idx + 1]           # (1,N)
        y_pred = np.repeat(y_anchor, H, axis=0)   # (H,N)

        err = (y_pred - y_true)                   # (H,N)

        # <3h
        err3 = err[:h3]
        m3 = m[:h3]
        abs_sum3 += float(np.sum(m3 * np.abs(err3)))
        sq_sum3  += float(np.sum(m3 * (err3 ** 2)))
        m_sum3   += float(np.sum(m3))

        # <6h
        err6 = err[:h6]
        m6 = m[:h6]
        abs_sum6 += float(np.sum(m6 * np.abs(err6)))
        sq_sum6  += float(np.sum(m6 * (err6 ** 2)))
        m_sum6   += float(np.sum(m6))

    mae3 = abs_sum3 / (m_sum3 + 1e-6)
    rmse3 = (sq_sum3 / (m_sum3 + 1e-6)) ** 0.5

    mae6 = abs_sum6 / (m_sum6 + 1e-6)
    rmse6 = (sq_sum6 / (m_sum6 + 1e-6)) ** 0.5

    # ---- denormalize to kW using y_sd ----
    y_sd = float(meta.get("y_sd", 1.0))
    mae3_kw, rmse3_kw = mae3 * y_sd, rmse3 * y_sd
    mae6_kw, rmse6_kw = mae6 * y_sd, rmse6 * y_sd

    print("==== Persistence baseline (anchor-hour, no future leakage) ====")
    print(f"root   = {root}")
    print(f"split  = {args.split}")
    print(f"L={L}, H={H}, ratio={ratio}, offset={offset}")
    print(f"samples used = {len(idxs)}")
    print(f"valid points (sum mask) <3h = {m_sum3:.0f} | <6h = {m_sum6:.0f}")
    print("")
    print(f"[NORMALIZED][<3h] MAE  = {mae3:.6f} | RMSE = {rmse3:.6f}")
    print(f"[NORMALIZED][<6h] MAE  = {mae6:.6f} | RMSE = {rmse6:.6f}")
    print("")
    print(f"[kW][<3h] MAE  = {mae3_kw:.3f} kW   ({mae3_kw/args.rated_kw*100:.2f}% of rated {args.rated_kw:g} kW)")
    print(f"[kW][<3h] RMSE = {rmse3_kw:.3f} kW  ({rmse3_kw/args.rated_kw*100:.2f}% of rated {args.rated_kw:g} kW)")
    print("")
    print(f"[kW][<6h] MAE  = {mae6_kw:.3f} kW   ({mae6_kw/args.rated_kw*100:.2f}% of rated {args.rated_kw:g} kW)")
    print(f"[kW][<6h] RMSE = {rmse6_kw:.3f} kW  ({rmse6_kw/args.rated_kw*100:.2f}% of rated {args.rated_kw:g} kW)")


if __name__ == "__main__":
    main()
