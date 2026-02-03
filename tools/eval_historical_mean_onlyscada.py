# tools/eval_historical_mean_onlyscada.py
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.wind_dataset3 import WindSTFDatasetOnlyScada5B


def build_key_id(time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    key = (month, day, hour) -> key_id in [0, 12*31*24)
    注意：对所有日期统一映射，月天数不足的日期不会出现，不影响。
    """
    month = time_index.month.values.astype(np.int32)  # 1..12
    day = time_index.day.values.astype(np.int32)      # 1..31
    hour = time_index.hour.values.astype(np.int32)    # 0..23
    key_id = (month - 1) * (31 * 24) + (day - 1) * 24 + hour
    return key_id.astype(np.int32)


def masked_mae_rmse(sum_abs: float, sum_sq: float, sum_m: float):
    mae = float(sum_abs / (sum_m + 1e-6))
    rmse = float(np.sqrt(sum_sq / (sum_m + 1e-6)))
    return mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/wind_onlyscada_no_5b")
    ap.add_argument("--split", type=str, default="test", choices=["test"])  # 只评估 test
    ap.add_argument("--history_years", type=str, default="pre2020",
                    choices=["pre2020", "train_only"],
                    help="pre2020=train+val(2016-2019), train_only=train(2016-2018)")
    ap.add_argument("--fallback", type=str, default="global_mean",
                    choices=["zero", "global_mean"],
                    help="if no historical samples for a (month,day,hour) key")
    ap.add_argument("--rated_kw", type=float, default=2050.0,
                    help="only used for printing % of rated (optional)")
    args = ap.parse_args()

    root = args.root
    meta = json.load(open(f"{root}/meta.json", "r"))
    splits = meta["splits"]

    Y = np.load(f"{root}/Y.npy").astype(np.float32)       # (T1, N) normalized
    M = np.load(f"{root}/mask.npy").astype(np.float32)    # (T1, N) 0/1
    T1, N = Y.shape

    H = int(meta.get("H", 6))
    L = int(meta.get("L", 9))

    h3 = min(3, H)
    h6 = min(6, H)

    # ---- build time index for every y_idx ----
    y_start = pd.to_datetime(meta["y_start"])
    y_step_h = int(meta["y_step_hours"])
    time_y = pd.date_range(y_start, periods=T1, freq=f"{y_step_h}h")
    key_id = build_key_id(time_y)  # (T1,)

    # ---- define historical range ----
    train_y0, train_y1 = map(int, splits["train_y"])
    val_y0, val_y1 = map(int, splits["val_y"])

    if args.history_years == "train_only":
        hist_y0, hist_y1 = train_y0, train_y1
    else:
        # pre2020: use train+val (up to val end)
        hist_y0, hist_y1 = 0, val_y1

    hist_y1 = min(hist_y1, T1)

    # ---- accumulate sums/counts per key, per node ----
    K = 12 * 31 * 24  # 8928
    sum_arr = np.zeros((K, N), dtype=np.float64)
    cnt_arr = np.zeros((K, N), dtype=np.float64)

    y_hist = Y[hist_y0:hist_y1]         # (Th, N)
    m_hist = M[hist_y0:hist_y1] > 0.5   # (Th, N)
    k_hist = key_id[hist_y0:hist_y1]    # (Th,)

    for n in range(N):
        valid_idx = np.where(m_hist[:, n])[0]
        if valid_idx.size == 0:
            continue
        kk = k_hist[valid_idx]
        vv = y_hist[valid_idx, n].astype(np.float64)
        np.add.at(sum_arr[:, n], kk, vv)
        np.add.at(cnt_arr[:, n], kk, 1.0)

    mean_arr = sum_arr / np.maximum(cnt_arr, 1.0)  # (K,N)

    # ---- fallback if cnt=0 ----
    if args.fallback == "zero":
        mean_arr[cnt_arr < 0.5] = 0.0
    elif args.fallback == "global_mean":
        gm = np.zeros((N,), dtype=np.float64)
        for n in range(N):
            vv = y_hist[:, n]
            mm = m_hist[:, n]
            gm[n] = float(vv[mm].mean()) if np.any(mm) else 0.0
        miss = (cnt_arr < 0.5)
        for n in range(N):
            mean_arr[miss[:, n], n] = gm[n]

    # ---- evaluate on EXACT same samples as your models: WindSTFDatasetOnlyScada5B(test) ----
    ds = WindSTFDatasetOnlyScada5B(root=root, split="test", L=L, H=H)
    ratio = int(ds.ratio)
    offset = int(ds.offset)

    # accumulators for <3h and <6h
    sum_abs3 = 0.0; sum_sq3 = 0.0; sum_m3 = 0.0
    sum_abs6 = 0.0; sum_sq6 = 0.0; sum_m6 = 0.0

    for x_idx in ds.idxs:
        y_idx = offset + x_idx * ratio

        for step in range(1, H + 1):
            t = y_idx + step
            if t >= T1:
                continue

            k = key_id[t]
            pred = mean_arr[k]                # (N,)
            truth = Y[t].astype(np.float64)   # (N,)
            mask = M[t].astype(np.float64)    # (N,)

            diff = pred - truth
            abs_term = float(np.sum(np.abs(diff) * mask))
            sq_term  = float(np.sum((diff * diff) * mask))
            m_term   = float(np.sum(mask))

            if step <= h6:
                sum_abs6 += abs_term
                sum_sq6  += sq_term
                sum_m6   += m_term
            if step <= h3:
                sum_abs3 += abs_term
                sum_sq3  += sq_term
                sum_m3   += m_term

    mae3, rmse3 = masked_mae_rmse(sum_abs3, sum_sq3, sum_m3)
    mae6, rmse6 = masked_mae_rmse(sum_abs6, sum_sq6, sum_m6)

    # optional: denormalize to kW
    y_sd = float(meta.get("y_sd", 1.0))
    mae3_kw, rmse3_kw = mae3 * y_sd, rmse3 * y_sd
    mae6_kw, rmse6_kw = mae6 * y_sd, rmse6 * y_sd

    print("=== Historical Mean Baseline (only_scada) ===")
    print(f"root = {root}")
    print(f"history_years = {args.history_years} | fallback = {args.fallback}")
    print(f"H={H} -> report <3h using first {h3} steps, <6h using first {h6} steps")
    print("")
    print(f"[TEST][NORMALIZED][<3h] MAE {mae3:.6f} | RMSE {rmse3:.6f}")
    print(f"[TEST][NORMALIZED][<6h] MAE {mae6:.6f} | RMSE {rmse6:.6f}")
    print("")
    print(f"[TEST][kW][<3h] MAE {mae3_kw:.3f} kW | RMSE {rmse3_kw:.3f} kW  ({rmse3_kw/args.rated_kw*100:.2f}% of rated)")
    print(f"[TEST][kW][<6h] MAE {mae6_kw:.3f} kW | RMSE {rmse6_kw:.3f} kW  ({rmse6_kw/args.rated_kw*100:.2f}% of rated)")
    print("")
    print("(mask-weighted; horizons are accumulated as your senior described)")


if __name__ == "__main__":
    main()
