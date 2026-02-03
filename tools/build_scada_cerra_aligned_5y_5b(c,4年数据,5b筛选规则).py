import os, re, json, glob
import numpy as np
import pandas as pd

RAW_DIR   = "raw_scada"
CERRA_DIR = "raw_new_cerra1hour"
OUT_DIR   = "data/wind_scada_cerra_5y"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_COL = "# Date and time"
PWR_COL  = "Power (kW)"
WSP_COL  = "Wind speed (m/s)"
WDIR_COL = "Wind direction (°)"
NAC_COL  = "Nacelle position (°)"
LOST_COL = "Lost Production to Downtime and Curtailment Total (kWh)"

X_STEP_HOURS = 3
Y_STEP_HOURS = 1
H = 6
L = 9

TRAIN_YEARS = {2016, 2017}
VAL_YEARS   = {2018}
TEST_YEARS  = {2019}
ALL_YEARS   = TRAIN_YEARS | VAL_YEARS | TEST_YEARS

DROP_JAN_1_2 = True

def turbine_id_from_filename(fname: str):
    m = re.search(r"Kelmarsh_(\d+)_", fname)
    return int(m.group(1)) if m else None

def read_one_scada_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=9)
    df.columns = [c.strip() for c in df.columns]
    need = {TIME_COL, PWR_COL, WSP_COL, WDIR_COL, NAC_COL, LOST_COL}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"{os.path.basename(path)} 缺少列: {sorted(miss)}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL)
    return df[[TIME_COL, PWR_COL, WSP_COL, WDIR_COL, NAC_COL, LOST_COL]]

def build_adj_from_static(static_path, turbine_ids):
    sdf = pd.read_csv(static_path)
    sdf.columns = [c.strip() for c in sdf.columns]
    name_col = "Title" if "Title" in sdf.columns else "Alternative Title"
    sdf["turbine_id"] = sdf[name_col].astype(str).str.extract(r"(\d+)").astype(int)
    sdf = sdf.rename(columns={"Latitude": "lat", "Longitude": "lon"})
    sdf = sdf[sdf["turbine_id"].isin(turbine_ids)].sort_values("turbine_id")

    coords = sdf[["lat","lon"]].values.astype(np.float32)
    N = coords.shape[0]
    lat0 = coords[:,0].mean()
    scale_lat = 111000.0
    scale_lon = 111000.0 * np.cos(np.deg2rad(lat0))
    xy = np.stack([(coords[:,1]-coords[:,1].mean())*scale_lon,
                   (coords[:,0]-coords[:,0].mean())*scale_lat], axis=1)

    d = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            d[i,j] = np.linalg.norm(xy[i]-xy[j])

    sigma = np.median(d[d>0]) if np.any(d>0) else 1.0
    A = np.exp(-(d**2)/(sigma**2)).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-6)
    return A

def is_ok_lost(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    ok = finite & (arr == 0)
    return ok.astype(np.float32)

def mask_drop_jan_1_2(time_index: pd.DatetimeIndex) -> np.ndarray:
    if not DROP_JAN_1_2:
        return np.ones((len(time_index),), dtype=np.float32)
    md = (time_index.month == 1) & (time_index.day <= 2)
    return (~md).astype(np.float32)

def idx_at(ts: pd.Timestamp, base: pd.Timestamp, step_h: int):
    return int((ts - base) / pd.Timedelta(hours=step_h))

def read_cerra_concat_for_turbine_75m(tid: int, full_time_x: pd.DatetimeIndex):
    tur_name = f"KWF{tid}"
    pattern = os.path.join(CERRA_DIR, f"CERRA_75m_{tur_name}_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"找不到 CERRA 文件: {pattern}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if "time" not in df.columns:
            raise KeyError(f"{p} 没有 time 列")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"])
    df_all = df_all[df_all["time"].dt.year.isin(ALL_YEARS)].copy()
    df_all = df_all.set_index("time")

    df_all = df_all.reindex(full_time_x)

    num_cols = [c for c in df_all.columns if c != "turbine"]
    sub = df_all[num_cols].copy()

    valid = np.isfinite(sub.values).astype(np.float32)
    sub = sub.fillna(0.0)

    sub.columns = [f"{c}_75" for c in sub.columns]
    return sub.values.astype(np.float32), valid.astype(np.float32), sub.columns.tolist()

def main():
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "Turbine_Data_Kelmarsh_*.csv")))
    if not all_files:
        raise FileNotFoundError(f"{RAW_DIR} 下找不到 Turbine_Data_Kelmarsh_*.csv")

    by_tid = {}
    for f in all_files:
        tid = turbine_id_from_filename(os.path.basename(f))
        if tid is None:
            continue
        by_tid.setdefault(tid, []).append(f)

    turbine_ids = sorted(by_tid.keys())
    if len(turbine_ids) != 6:
        raise RuntimeError(f"期望6个风机文件组，但解析到 {len(turbine_ids)} 个: {turbine_ids}")

    turb_raw = {}
    for tid in turbine_ids:
        dfs = []
        for f in sorted(by_tid[tid]):
            df = read_one_scada_file(f)
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=[TIME_COL]).sort_values(TIME_COL)
        df_all = df_all[df_all[TIME_COL].dt.year.isin(ALL_YEARS)].copy()
        turb_raw[tid] = df_all

    t0 = min(df[TIME_COL].min() for df in turb_raw.values())
    t1 = max(df[TIME_COL].max() for df in turb_raw.values())

    t_y0 = pd.to_datetime(t0).ceil("1h")
    t_y1 = pd.to_datetime(t1).floor("1h")
    full_time_y = pd.date_range(t_y0, t_y1, freq="1h")

    t_x0 = t_y0.ceil(f"{X_STEP_HOURS}h")
    t_x1 = t_y1.floor(f"{X_STEP_HOURS}h")
    full_time_x = pd.date_range(t_x0, t_x1, freq=f"{X_STEP_HOURS}h")

    ratio = X_STEP_HOURS // Y_STEP_HOURS
    offset_hours = int((t_x0 - t_y0) / pd.Timedelta(hours=1))
    if ratio != 3:
        raise RuntimeError("当前实现假设 X_STEP_HOURS=3, Y_STEP_HOURS=1")

    x_2018 = idx_at(pd.Timestamp("2018-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    x_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    y_2018 = idx_at(pd.Timestamp("2018-01-01 00:00:00"), t_y0, Y_STEP_HOURS)
    y_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_y0, Y_STEP_HOURS)


    keep_y = mask_drop_jan_1_2(full_time_y)
    keep_x = mask_drop_jan_1_2(full_time_x)

    P1_list, M1_list = [], []
    X_list, Xv_list = [], []
    cerra_names = None

    for tid in turbine_ids:
        df = turb_raw[tid]
        df_hp = df[df[TIME_COL].dt.minute.eq(0) & df[TIME_COL].dt.second.eq(0)].copy()
        df_hp = df_hp.groupby(TIME_COL).mean(numeric_only=True)
        df_hp = df_hp.reindex(full_time_y)

        P1_raw   = df_hp[PWR_COL].to_numpy(dtype=np.float32)
        W1_raw   = df_hp[WSP_COL].to_numpy(dtype=np.float32)
        Dir1_raw = df_hp[WDIR_COL].to_numpy(dtype=np.float32)
        Nac1_raw = df_hp[NAC_COL].to_numpy(dtype=np.float32)

        dfl = df.set_index(TIME_COL)[[LOST_COL]].sort_index()
        lost1 = dfl.resample("1h").sum(min_count=1).reindex(full_time_y)[LOST_COL].to_numpy(dtype=np.float32)

        ok_lost1 = is_ok_lost(lost1) > 0.5
        p1_finite = np.isfinite(P1_raw)
        mask_y = (p1_finite & ok_lost1 & (keep_y > 0.5)).astype(np.float32)

        P1 = np.nan_to_num(P1_raw, nan=0.0)
        P1[mask_y < 0.5] = 0.0
        P1_list.append(P1)
        M1_list.append(mask_y)

        P3_raw   = pd.Series(P1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        W3_raw   = pd.Series(W1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Dir3_raw = pd.Series(Dir1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Nac3_raw = pd.Series(Nac1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)

        lost3 = dfl.resample(f"{X_STEP_HOURS}h").sum(min_count=1).reindex(full_time_x)[LOST_COL].to_numpy(dtype=np.float32)
        ok_lost3 = is_ok_lost(lost3) > 0.5

        p3_valid = (np.isfinite(P3_raw) & ok_lost3 & (keep_x > 0.5)).astype(np.float32)
        w3_valid   = (np.isfinite(W3_raw)   & (keep_x > 0.5)).astype(np.float32)
        dir_valid  = (np.isfinite(Dir3_raw) & (keep_x > 0.5)).astype(np.float32)
        nac_valid  = (np.isfinite(Nac3_raw) & (keep_x > 0.5)).astype(np.float32)

        P3 = np.nan_to_num(P3_raw, nan=0.0)
        W3 = np.nan_to_num(W3_raw, nan=0.0)

        dir_rad = np.deg2rad(np.nan_to_num(Dir3_raw, nan=0.0))
        nac_rad = np.deg2rad(np.nan_to_num(Nac3_raw, nan=0.0))
        dir_sin = np.sin(dir_rad).astype(np.float32)
        dir_cos = np.cos(dir_rad).astype(np.float32)
        nac_sin = np.sin(nac_rad).astype(np.float32)
        nac_cos = np.cos(nac_rad).astype(np.float32)

        dir_sin[dir_valid < 0.5] = 0.0
        dir_cos[dir_valid < 0.5] = 0.0
        nac_sin[nac_valid < 0.5] = 0.0
        nac_cos[nac_valid < 0.5] = 0.0

        dP = np.diff(P3, axis=0, prepend=P3[[0]])
        prev = np.roll(p3_valid, 1); prev[0] = 0.0
        dp_valid = ((p3_valid > 0.5) & (prev > 0.5)).astype(np.float32)
        dP[dp_valid < 0.5] = 0.0

        P3[p3_valid < 0.5] = 0.0

        X_scada = np.stack([P3, dP, W3, dir_sin, dir_cos, nac_sin, nac_cos], axis=-1).astype(np.float32)

        Xv_scada = np.zeros_like(X_scada, dtype=np.float32)
        Xv_scada[..., 0] = p3_valid
        Xv_scada[..., 1] = dp_valid
        Xv_scada[..., 2] = w3_valid
        Xv_scada[..., 3] = dir_valid
        Xv_scada[..., 4] = dir_valid
        Xv_scada[..., 5] = nac_valid
        Xv_scada[..., 6] = nac_valid

        C, Cv, names = read_cerra_concat_for_turbine_75m(tid, full_time_x)
        if cerra_names is None:
            cerra_names = names

        keep_x_col = keep_x.reshape(-1, 1).astype(np.float32)
        Cv = Cv * keep_x_col

        X_all = np.concatenate([X_scada, C], axis=-1).astype(np.float32)
        Xv_all = np.concatenate([Xv_scada, Cv], axis=-1).astype(np.float32)

        X_list.append(X_all)
        Xv_list.append(Xv_all)

    X_raw = np.stack(X_list, axis=1).astype(np.float32)      # (T3,N,F)
    X_valid = np.stack(Xv_list, axis=1).astype(np.float32)   # (T3,N,F)
    Y_raw = np.stack(P1_list, axis=1).astype(np.float32)     # (T1,N)
    mask_y = np.stack(M1_list, axis=1).astype(np.float32)    # (T1,N)

    T3 = X_raw.shape[0]
    T1 = Y_raw.shape[0]

    x_train_end = max(0, min(x_2018, T3))
    y_train_end = max(0, min(y_2018, T1))


    F = X_raw.shape[-1]
    x_mu = np.zeros((F,), dtype=np.float32)
    x_sd = np.ones((F,), dtype=np.float32)

    Xtr = X_raw[:x_train_end]
    Vtr = X_valid[:x_train_end] > 0.5
    for f in range(F):
        v = Xtr[..., f]
        m = Vtr[..., f]
        if np.any(m):
            vals = v[m]
            x_mu[f] = float(vals.mean())
            x_sd[f] = float(vals.std()) + 1e-6
        else:
            x_mu[f] = 0.0
            x_sd[f] = 1.0

    Ytr = Y_raw[:y_train_end]
    Mtr = mask_y[:y_train_end] > 0.5
    if np.any(Mtr):
        y_mu = float(Ytr[Mtr].mean())
        y_sd = float(Ytr[Mtr].std()) + 1e-6
    else:
        y_mu, y_sd = 0.0, 1.0

    Xn = (X_raw - x_mu) / x_sd
    Yn = (Y_raw - y_mu) / y_sd

    Xn[X_valid < 0.5] = 0.0
    Yn[mask_y < 0.5] = 0.0

    static_path = os.path.join(RAW_DIR, "Kelmarsh_WT_static.csv")
    if os.path.exists(static_path):
        A = build_adj_from_static(static_path, turbine_ids)
    else:
        N = len(turbine_ids)
        A = np.ones((N, N), dtype=np.float32) / N

    np.save(f"{OUT_DIR}/X.npy", Xn.astype(np.float32))
    np.save(f"{OUT_DIR}/X_valid.npy", X_valid.astype(np.float32))
    np.save(f"{OUT_DIR}/Y.npy", Yn.astype(np.float32))
    np.save(f"{OUT_DIR}/mask.npy", mask_y.astype(np.float32))
    np.save(f"{OUT_DIR}/adj.npy", A.astype(np.float32))

    feature_names = ["P3","dP3","W3","dir_sin3","dir_cos3","nac_sin3","nac_cos3"] + (cerra_names if cerra_names else [])

    meta = {
        "turbine_ids": turbine_ids,
        "feature_names": feature_names,
        "x_mu": x_mu.tolist(),
        "x_sd": x_sd.tolist(),
        "y_mu": float(y_mu),
        "y_sd": float(y_sd),

        "L": int(L),
        "H": int(H),
        "x_step_hours": int(X_STEP_HOURS),
        "y_step_hours": int(Y_STEP_HOURS),
        "ratio_x_to_y": int(ratio),
        "offset_hours": int(offset_hours),

        "lost_col": LOST_COL,
        "lost_rule": {"NaN":"exclude","0":"include",">0":"exclude"},
        "drop_jan_1_2": bool(DROP_JAN_1_2),

        "x_start": str(t_x0),
        "x_end":   str(t_x1),
        "y_start": str(t_y0),
        "y_end":   str(t_y1),

        "splits": {
            "train_x": [0, int(x_2018)],
            "val_x":   [int(x_2018), int(x_2019)],
            "test_x":  [int(x_2019), int(T3)],

            "train_y": [0, int(y_2018)],
            "val_y":   [int(y_2018), int(y_2019)],
            "test_y":  [int(y_2019), int(T1)],
        }

    }

    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[OK] Saved to", OUT_DIR)
    print("X", Xn.shape, "X_valid", X_valid.shape, "Y", Yn.shape, "mask", mask_y.shape, "adj", A.shape)
    print("Features:", len(feature_names))
    print(f"X time: {t_x0} -> {t_x1} (freq=3h), len={len(full_time_x)}")
    print(f"Y time: {t_y0} -> {t_y1} (freq=1h), len={len(full_time_y)}")
    print("Split X idx:", meta["splits"]["train_x"], meta["splits"]["val_x"], meta["splits"]["test_x"])

if __name__ == "__main__":
    main()
