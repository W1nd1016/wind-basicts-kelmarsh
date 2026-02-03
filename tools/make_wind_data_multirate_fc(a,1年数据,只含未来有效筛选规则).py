import os, re, json, glob
import numpy as np
import pandas as pd

RAW_DIR   = "raw_scada"
CERRA_DIR = "raw_new_cerra1hour"
OUT_DIR   = "data/wind4_1hour"
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

HEIGHT_LEVELS = [75]
SCADA_YEAR = "2016"

def read_one_turbine_csv(path):
    df = pd.read_csv(path, skiprows=9)
    df.columns = [c.strip() for c in df.columns]
    assert LOST_COL in df.columns, f"{os.path.basename(path)} 缺少列: {LOST_COL}"
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL)
    keep = [TIME_COL, PWR_COL, WSP_COL, WDIR_COL, NAC_COL, LOST_COL]
    return df[keep]

def turbine_id_from_filename(fname):
    m = re.search(r"Kelmarsh_(\d+)_", fname)
    return int(m.group(1)) if m else None

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

def read_cerra_for_turbine_75m(tid, full_time_x):
    tur_name = f"KWF{tid}"
    pattern = os.path.join(CERRA_DIR, f"CERRA_75m_{tur_name}.csv")
    paths = sorted(glob.glob(pattern))
    assert paths, f"找不到 CERRA 文件: {pattern}"
    path = paths[0]
    print(f"[CERRA] tid={tid} -> {os.path.basename(path)}")

    df = pd.read_csv(path)
    assert "time" in df.columns, f"{path} 没有 time 列"
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")

    df = df.reindex(full_time_x)

    num_cols = [c for c in df.columns if c != "turbine"]
    sub = df[num_cols].copy()

    valid = np.isfinite(sub.values).astype(np.float32)
    sub = sub.fillna(0.0)

    sub.columns = [f"{c}_75" for c in sub.columns]
    return sub.values.astype(np.float32), valid.astype(np.float32), sub.columns.tolist()

def is_ok_lost(arr):
    finite = np.isfinite(arr)
    ok = finite & (arr == 0)
    return ok.astype(np.float32)

def main():
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "Turbine_Data_Kelmarsh_*.csv")))
    files = [f for f in all_files if f"_{SCADA_YEAR}-" in os.path.basename(f)]
    assert len(files) == 6, f"Expect 6 turbine files for {SCADA_YEAR}, got {len(files)}"

    turb_dfs = []
    turb_ids = []
    for f in files:
        tid = turbine_id_from_filename(os.path.basename(f))
        assert tid is not None, f"无法从文件名解析tid: {f}"
        turb_ids.append(tid)
        df = read_one_turbine_csv(f)
        df["turbine_id"] = tid
        turb_dfs.append(df)

    turb_ids = sorted(turb_ids)
    turb_dfs = [df for _, df in sorted(
        zip([df["turbine_id"].iloc[0] for df in turb_dfs], turb_dfs)
    )]

    hourly_list = []
    for df in turb_dfs:
        tid = df["turbine_id"].iloc[0]
        dft = df.set_index(TIME_COL)

        d1_mean = dft[[PWR_COL, WSP_COL, WDIR_COL, NAC_COL]].resample("1h").mean(numeric_only=True)
        d1_lost = dft[[LOST_COL]].resample("1h").sum(min_count=1)
        d1 = pd.concat([d1_mean, d1_lost], axis=1)
        d1["turbine_id"] = tid
        hourly_list.append(d1.reset_index())

    hdf1 = pd.concat(hourly_list, ignore_index=True)

    t_y0 = pd.to_datetime(hdf1[TIME_COL].min()).ceil("1h")
    t_y1 = pd.to_datetime(hdf1[TIME_COL].max()).floor("1h")
    full_time_y = pd.date_range(t_y0, t_y1, freq="1h")

    t_x0 = t_y0.ceil(f"{X_STEP_HOURS}h")
    t_x1 = t_y1.floor(f"{X_STEP_HOURS}h")
    full_time_x = pd.date_range(t_x0, t_x1, freq=f"{X_STEP_HOURS}h")

    ratio = X_STEP_HOURS // Y_STEP_HOURS
    offset_hours = int((t_x0 - t_y0) / pd.Timedelta(hours=1))
    assert ratio == 3, "当前实现假设 X=3h, Y=1h"

    P1_list, M1_list = [], []
    X_scada_list = []
    X_scada_valid_list = []

    C_feat_list = []
    C_valid_list = []
    cerra_feature_names = None

    for df in turb_dfs:
        tid = int(df["turbine_id"].iloc[0])
        dft = df.set_index(TIME_COL)

        d1_pwr  = dft[[PWR_COL]].resample("1h").mean(numeric_only=True).reindex(full_time_y)
        d1_lost = dft[[LOST_COL]].resample("1h").sum(min_count=1).reindex(full_time_y)

        P1_raw = d1_pwr[PWR_COL].values.astype(np.float32)
        lost1  = d1_lost[LOST_COL].values.astype(np.float32)

        p1_finite = np.isfinite(P1_raw)
        ok_lost1  = is_ok_lost(lost1) > 0.5
        mask_y = (p1_finite & ok_lost1).astype(np.float32)

        P1 = np.nan_to_num(P1_raw, nan=0.0)
        P1[mask_y < 0.5] = 0.0

        P1_list.append(P1)
        M1_list.append(mask_y)

        d3_mean = dft[[PWR_COL, WSP_COL, WDIR_COL, NAC_COL]].resample(f"{X_STEP_HOURS}h").mean(numeric_only=True).reindex(full_time_x)
        d3_lost = dft[[LOST_COL]].resample(f"{X_STEP_HOURS}h").sum(min_count=1).reindex(full_time_x)

        P3_raw   = d3_mean[PWR_COL].values.astype(np.float32)
        W3_raw   = d3_mean[WSP_COL].values.astype(np.float32)
        Dir3_raw = d3_mean[WDIR_COL].values.astype(np.float32)
        Nac3_raw = d3_mean[NAC_COL].values.astype(np.float32)
        lost3    = d3_lost[LOST_COL].values.astype(np.float32)

        ok_lost3 = is_ok_lost(lost3) > 0.5

        p3_valid = (np.isfinite(P3_raw) & ok_lost3).astype(np.float32)

        w3_valid   = np.isfinite(W3_raw).astype(np.float32)
        dir_valid  = np.isfinite(Dir3_raw).astype(np.float32)
        nac_valid  = np.isfinite(Nac3_raw).astype(np.float32)

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
        X_scada_list.append(X_scada)

        Xv = np.zeros_like(X_scada, dtype=np.float32)
        Xv[..., 0] = p3_valid
        Xv[..., 1] = dp_valid
        Xv[..., 2] = w3_valid
        Xv[..., 3] = dir_valid
        Xv[..., 4] = dir_valid
        Xv[..., 5] = nac_valid
        Xv[..., 6] = nac_valid
        X_scada_valid_list.append(Xv)

        C, Cv, names = read_cerra_for_turbine_75m(tid, full_time_x)
        if cerra_feature_names is None:
            cerra_feature_names = names
        C_feat_list.append(C)
        C_valid_list.append(Cv)

    X_scada = np.stack(X_scada_list, axis=1).astype(np.float32)
    X_scada_valid = np.stack(X_scada_valid_list, axis=1).astype(np.float32)

    C_feat = np.stack(C_feat_list, axis=1).astype(np.float32)
    C_valid = np.stack(C_valid_list, axis=1).astype(np.float32)

    X_raw = np.concatenate([X_scada, C_feat], axis=-1).astype(np.float32)
    X_valid = np.concatenate([X_scada_valid, C_valid], axis=-1).astype(np.float32)

    Y_raw = np.stack(P1_list, axis=1).astype(np.float32)
    mask_y = np.stack(M1_list, axis=1).astype(np.float32)

    T3 = X_raw.shape[0]
    t_train = int(T3 * 0.8)
    t_val   = int(T3 * 0.9)

    F = X_raw.shape[-1]
    x_mu = np.zeros((F,), dtype=np.float32)
    x_sd = np.ones((F,), dtype=np.float32)

    Xtr = X_raw[:t_train]
    Mtr = X_valid[:t_train] > 0.5
    for f in range(F):
        v = Xtr[..., f]
        m = Mtr[..., f]
        if np.any(m):
            vals = v[m]
            x_mu[f] = float(vals.mean())
            x_sd[f] = float(vals.std()) + 1e-6
        else:
            x_mu[f] = 0.0
            x_sd[f] = 1.0

    train_y_end = offset_hours + (t_train - 1) * ratio + H
    train_y_end = min(train_y_end, Y_raw.shape[0] - 1)

    Ytr = Y_raw[:train_y_end+1]
    Mtr_y = mask_y[:train_y_end+1] > 0.5
    if np.any(Mtr_y):
        y_mu = float(Ytr[Mtr_y].mean())
        y_sd = float(Ytr[Mtr_y].std()) + 1e-6
    else:
        y_mu, y_sd = 0.0, 1.0

    Xn = (X_raw - x_mu) / x_sd
    Yn = (Y_raw - y_mu) / y_sd

    Xn[X_valid < 0.5] = 0.0
    Yn[mask_y < 0.5] = 0.0

    static_path = os.path.join(RAW_DIR, "Kelmarsh_WT_static.csv")
    if os.path.exists(static_path):
        A = build_adj_from_static(static_path, turb_ids)
    else:
        N = len(turb_ids)
        A = np.ones((N, N), dtype=np.float32) / N

    np.save(f"{OUT_DIR}/X.npy", Xn.astype(np.float32))
    np.save(f"{OUT_DIR}/Y.npy", Yn.astype(np.float32))
    np.save(f"{OUT_DIR}/mask.npy", mask_y.astype(np.float32))
    np.save(f"{OUT_DIR}/adj.npy", A.astype(np.float32))

    feature_names = [
        "P3", "dP3", "W3",
        "dir_sin3", "dir_cos3",
        "nac_sin3", "nac_cos3",
    ] + (cerra_feature_names if cerra_feature_names else [])

    meta = {
        "turbine_ids": turb_ids,
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
        "lost_rule": {
            "NaN": "exclude",
            "0": "include",
            ">0": "exclude"
        },
        "x_start": str(t_x0),
        "x_end":   str(t_x1),
        "y_start": str(t_y0),
        "y_end":   str(t_y1),
        "splits": {
            "train_x": [0, t_train],
            "val_x":   [t_train, t_val],
            "test_x":  [t_val, T3]
        }
    }
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved to", OUT_DIR)
    print("X", Xn.shape, "Y", Yn.shape, "mask(hourly)", mask_y.shape, "adj", A.shape)
    print("Features:", len(feature_names))
    print(f"X time: {t_x0} -> {t_x1} (freq=3h), len={len(full_time_x)}")
    print(f"Y time: {t_y0} -> {t_y1} (freq=1h), len={len(full_time_y)}")
    print(f"Split X idx: train[0,{t_train}) val[{t_train},{t_val}) test[{t_val},{T3})")

if __name__ == "__main__":
    main()
