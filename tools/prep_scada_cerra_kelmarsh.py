import os, re, json, glob
import numpy as np
import pandas as pd

RAW_DIR   = "raw_scada"
CERRA_DIR = "raw_cerra"
OUT_DIR   = "data/wind"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_COL = "# Date and time"
PWR_COL  = "Power (kW)"
WSP_COL  = "Wind speed (m/s)"
WDIR_COL = "Wind direction (°)"
NAC_COL  = "Nacelle position (°)"

L = 24
H = 6

HEIGHT_LEVELS = [50, 75, 100]

def read_one_turbine_csv(path):
    df = pd.read_csv(path, skiprows=9)
    df.columns = [c.strip() for c in df.columns]
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL)
    keep = [TIME_COL, PWR_COL, WSP_COL, WDIR_COL, NAC_COL]
    df = df[keep]
    return df

def turbine_id_from_filename(fname):
    m = re.search(r"Kelmarsh_(\d+)_", fname)
    if m:
        return int(m.group(1))
    return None

def build_adj_from_static(static_path, turbine_ids):
    sdf = pd.read_csv(static_path)
    sdf.columns = [c.strip() for c in sdf.columns]

    if "Title" in sdf.columns:
        name_col = "Title"
    else:
        name_col = "Alternative Title"

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

def read_cerra_for_turbine_all_levels(tid, full_time):
    dfs = []
    for lvl in HEIGHT_LEVELS:
        tur_name = f"KWF{tid}"
        pattern = os.path.join(CERRA_DIR, f"CERRA_{lvl}m_{tur_name}.csv")
        paths = sorted(glob.glob(pattern))
        assert paths, f"找不到 CERRA 文件: {pattern}"
        path = paths[0]
        print(f"[CERRA] tid={tid} level={lvl} -> {os.path.basename(path)}")

        df = pd.read_csv(path)
        assert "time" in df.columns, f"{path} 没有 time 列"
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").set_index("time")

        df_3h = df.asfreq("3h")
        df_1h = df_3h.asfreq("1h").interpolate()
        df_1h = df_1h.reindex(full_time).interpolate().ffill().bfill()

        num_cols = [c for c in df_1h.columns if c != "turbine"]
        sub = df_1h[num_cols].copy()
        sub.columns = [f"{c}_{lvl}" for c in num_cols]

        dfs.append(sub)

    df_all = pd.concat(dfs, axis=1)
    return df_all

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "Turbine_Data_Kelmarsh_*.csv")))
    assert len(files) == 6, f"Expect 6 turbine files, got {len(files)}"

    turb_dfs = []
    turb_ids = []

    for f in files:
        tid = turbine_id_from_filename(os.path.basename(f))
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
        h = dft.resample("1h").mean(numeric_only=True)
        h["turbine_id"] = tid
        hourly_list.append(h.reset_index())

    hdf = pd.concat(hourly_list, ignore_index=True)

    full_time = pd.date_range(hdf[TIME_COL].min(), hdf[TIME_COL].max(), freq="1h")

    P_list, W_list, Dir_list, Nac_list, M_list = [], [], [], [], []
    C_feat_list = []
    cerra_feature_names = None

    for tid in turb_ids:
        dft = hdf[hdf["turbine_id"] == tid].set_index(TIME_COL).reindex(full_time)
        P_list.append(dft[PWR_COL].values)
        W_list.append(dft[WSP_COL].values)
        Dir_list.append(dft[WDIR_COL].values)
        Nac_list.append(dft[NAC_COL].values)
        M_list.append(np.isfinite(dft[PWR_COL].values).astype(np.float32))

        cerra_df = read_cerra_for_turbine_all_levels(tid, full_time)
        if cerra_feature_names is None:
            cerra_feature_names = cerra_df.columns.tolist()
        C_feat_list.append(cerra_df.values)

    P   = np.stack(P_list, axis=1)
    W   = np.stack(W_list, axis=1)
    Dir = np.stack(Dir_list, axis=1)
    Nac = np.stack(Nac_list, axis=1)
    mask = np.stack(M_list, axis=1)

    C_feat = np.stack(C_feat_list, axis=1).astype(np.float32)

    P   = pd.DataFrame(P).ffill().bfill().fillna(0).values
    W   = pd.DataFrame(W).ffill().bfill().fillna(0).values
    Dir = pd.DataFrame(Dir).ffill().bfill().fillna(0).values
    Nac = pd.DataFrame(Nac).ffill().bfill().fillna(0).values

    dir_rad = np.deg2rad(Dir)
    nac_rad = np.deg2rad(Nac)
    dir_sin, dir_cos = np.sin(dir_rad), np.cos(dir_rad)
    nac_sin, nac_cos = np.sin(nac_rad), np.cos(nac_rad)

    dP = np.diff(P, axis=0, prepend=P[[0], :])

    X_scada = np.stack(
        [P, dP, W,
         dir_sin, dir_cos,
         nac_sin, nac_cos],
        axis=-1
    ).astype(np.float32)

    X = np.concatenate([X_scada, C_feat], axis=-1)
    Y = P.astype(np.float32)

    T_all = X.shape[0]
    t_train = int(T_all * 0.8)
    t_val   = int(T_all * 0.9)

    x_mu = X[:t_train].reshape(-1, X.shape[-1]).mean(axis=0)
    x_sd = X[:t_train].reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    y_mu = Y[:t_train].mean()
    y_sd = Y[:t_train].std() + 1e-6

    Xn = (X - x_mu) / x_sd
    Yn = (Y - y_mu) / y_sd

    static_path = os.path.join(RAW_DIR, "Kelmarsh_WT_static.csv")
    if os.path.exists(static_path):
        A = build_adj_from_static(static_path, turb_ids)
    else:
        N = len(turb_ids)
        A = np.ones((N, N), dtype=np.float32) / N

    np.save(f"{OUT_DIR}/X.npy", Xn)
    np.save(f"{OUT_DIR}/Y.npy", Yn)
    np.save(f"{OUT_DIR}/mask.npy", mask.astype(np.float32))
    np.save(f"{OUT_DIR}/adj.npy", A)

    feature_names = [
        "P", "dP", "W",
        "dir_sin", "dir_cos",
        "nac_sin", "nac_cos",
    ] + cerra_feature_names # type: ignore

    meta = {
        "turbine_ids": turb_ids,
        "x_mu": x_mu.tolist(),
        "x_sd": x_sd.tolist(),
        "y_mu": float(y_mu),
        "y_sd": float(y_sd),
        "splits": {
            "train": [0, t_train],
            "val":   [t_train, t_val],
            "test":  [t_val, T_all]
        },
        "feature_names": feature_names,
        "history_len": L,
        "horizon": H,
        "L": L,
        "H": H
    }
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved to", OUT_DIR)
    print("X", Xn.shape, "Y", Yn.shape, "mask", mask.shape, "adj", A.shape)
    print("Features:", len(feature_names))

if __name__ == "__main__":
    main()