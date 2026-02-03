# tools/build_scada_cerra_v1_s2.py
import os, re, json, glob
import numpy as np
import pandas as pd

RAW_DIR   = "raw_scada"
CERRA_DIR = "raw_new_cerra1hour"
OUT_DIR   = "data/wind_scada_cerra_v1_s2"
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

TRAIN_YEARS = {2016, 2017, 2018}
VAL_YEARS   = {2019}
TEST_YEARS  = {2020}
ALL_YEARS   = TRAIN_YEARS | VAL_YEARS | TEST_YEARS

HEIGHT_LEVELS = [75]
FORECAST_LEADS = [1,2,3,4,5,6]
K_NEIGHBORS = 16

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

def is_ok_lost(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    ok = finite & (arr == 0)
    return ok.astype(np.float32)

def build_adj_and_xy_from_static(static_path, turbine_ids):
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
    xy = np.stack([
        (coords[:,1]-coords[:,1].mean())*scale_lon,
        (coords[:,0]-coords[:,0].mean())*scale_lat
    ], axis=1).astype(np.float32)

    d = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            d[i,j] = np.linalg.norm(xy[i]-xy[j])

    sigma = np.median(d[d>0]) if np.any(d>0) else 1.0
    A = np.exp(-(d**2)/(sigma**2)).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-6)

    xy_norm = xy.copy()
    s = np.std(xy_norm, axis=0, keepdims=True) + 1e-6
    xy_norm = xy_norm / s
    return A.astype(np.float32), coords.astype(np.float32), xy_norm.astype(np.float32)

def parse_year_from_cerra_fname(fname: str):
    m = re.search(r"_(\d{4})\.csv$", fname)
    return int(m.group(1)) if m else None

def read_cerra_for_turbine_all_years(tid: int, full_time_x: pd.DatetimeIndex):
    """
    读 CERRA_75m_KWF{tid}_{year}.csv
    返回:
      sub_vals: (T3, Fc)
      valid   : (T3, Fc)
      colnames: list[str]  (已加 _75)
    """
    tur_name = f"KWF{tid}"
    pattern_year = os.path.join(CERRA_DIR, f"CERRA_75m_{tur_name}_*.csv")
    paths_year = sorted(glob.glob(pattern_year))
    if not paths_year:
        raise FileNotFoundError(f"找不到 CERRA 年文件: {pattern_year}")

    dfs = []
    for p in paths_year:
        y = parse_year_from_cerra_fname(os.path.basename(p))
        if y is None or y not in ALL_YEARS:
            continue
        df = pd.read_csv(p)
        if "time" not in df.columns:
            raise KeyError(f"{p} 没有 time 列")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"{tur_name} 没有任何 {sorted(ALL_YEARS)} 年份的 CERRA 数据")

    df_all = pd.concat(dfs, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"]).set_index("time")

    num_cols = [c for c in df_all.columns if c not in ["turbine", "time"]]
    sub = df_all[num_cols].copy()

    # 对齐到 full_time_x（3h），缺失 NaN
    sub = sub.reindex(full_time_x)

    valid = np.isfinite(sub.values).astype(np.float32)
    sub = sub.fillna(0.0)

    sub.columns = [f"{c}_75" for c in sub.columns]
    return sub.values.astype(np.float32), valid.astype(np.float32), sub.columns.tolist()

def idx_at(ts: pd.Timestamp, base: pd.Timestamp, step_h: int) -> int:
    return int((ts - base) / pd.Timedelta(hours=step_h))

def load_neighbors_coords(tid: int):
    """
    读取 raw_new_cerra1hour/CERRA_75m_KWF{tid}_neighbors.npz
    返回:
      coords_feat: (3K,)  (dx,dy,dist) 已稳定缩放
      names: list[str] with _75 suffix
    """
    npz_path = os.path.join(CERRA_DIR, f"CERRA_75m_KWF{tid}_neighbors.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing neighbors npz: {npz_path}")

    z = np.load(npz_path, allow_pickle=True)
    dx_m = z["dx_m"].astype(np.float32)     # (K,)
    dy_m = z["dy_m"].astype(np.float32)
    dist_m = z["dist_m"].astype(np.float32)

    dx_scale = float(np.max(np.abs(dx_m)) + 1e-6)
    dy_scale = float(np.max(np.abs(dy_m)) + 1e-6)
    dist_scale = float(np.max(dist_m) + 1e-6)

    dx = (dx_m / dx_scale).astype(np.float32)
    dy = (dy_m / dy_scale).astype(np.float32)
    ds = (dist_m / dist_scale).astype(np.float32)

    feat = np.concatenate([dx, dy, ds], axis=0).astype(np.float32)  # (3K,)
    K = dx.shape[0]
    names = [f"dx_k{i+1}_75" for i in range(K)] + \
            [f"dy_k{i+1}_75" for i in range(K)] + \
            [f"dist_k{i+1}_75" for i in range(K)]
    return feat, names

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
        raise RuntimeError(f"期望6台风机，但解析到 {len(turbine_ids)} 台: {turbine_ids}")

    turb_raw = {}
    for tid in turbine_ids:
        dfs = []
        for f in sorted(by_tid[tid]):
            dfs.append(read_one_scada_file(f))
        df_all = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=[TIME_COL]).sort_values(TIME_COL)
        df_all = df_all[df_all[TIME_COL].dt.year.isin(ALL_YEARS)].copy()
        turb_raw[tid] = df_all

    # -------- time axis ----------
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

    x_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    x_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    y_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_y0, Y_STEP_HOURS)
    y_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_y0, Y_STEP_HOURS)

    # -------- build arrays ----------
    P1_list, M1_list = [], []
    X_list, Xv_list = [], []
    C_feat_list, C_valid_list = [], []
    cerra_feature_names = None

    for tid in turbine_ids:
        df = turb_raw[tid]
        dft = df.set_index(TIME_COL).sort_index()

        sub_hour = dft[[PWR_COL, WSP_COL, WDIR_COL, NAC_COL]].copy()
        sub_hour = sub_hour[(sub_hour.index.minute == 0) & (sub_hour.index.second == 0)]
        sub_hour = sub_hour.groupby(level=0).mean(numeric_only=True)
        sub_hour = sub_hour.reindex(full_time_y)

        d1_lost = dft[[LOST_COL]].resample("1h").sum(min_count=1).reindex(full_time_y)

        P1_raw = sub_hour[PWR_COL].to_numpy(dtype=np.float32)
        lost1  = d1_lost[LOST_COL].to_numpy(dtype=np.float32)

        p1_finite = np.isfinite(P1_raw)
        ok_lost1  = is_ok_lost(lost1) > 0.5
        mask_y = (p1_finite & ok_lost1).astype(np.float32)

        P1 = np.nan_to_num(P1_raw, nan=0.0).astype(np.float32)
        P1[mask_y < 0.5] = 0.0
        P1_list.append(P1)
        M1_list.append(mask_y)

        W1_raw   = sub_hour[WSP_COL].to_numpy(dtype=np.float32)
        Dir1_raw = sub_hour[WDIR_COL].to_numpy(dtype=np.float32)
        Nac1_raw = sub_hour[NAC_COL].to_numpy(dtype=np.float32)

        P3_raw   = pd.Series(P1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        W3_raw   = pd.Series(W1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Dir3_raw = pd.Series(Dir1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Nac3_raw = pd.Series(Nac1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)

        lost1_s = pd.Series(lost1, index=full_time_y)
        finite = np.isfinite(lost1_s.to_numpy(dtype=np.float64)).astype(np.float64)
        s0 = pd.Series(np.nan_to_num(lost1_s.to_numpy(dtype=np.float64), nan=0.0), index=full_time_y)

        sum3 = s0.rolling(window=3, min_periods=3).sum()
        cnt3 = pd.Series(finite, index=full_time_y).rolling(window=3, min_periods=3).sum()
        lost3_s = sum3.where(cnt3 == 3.0, np.nan)
        lost3 = lost3_s.reindex(full_time_x).to_numpy(dtype=np.float32)

        ok_lost3 = is_ok_lost(lost3) > 0.5

        p3_valid  = (np.isfinite(P3_raw) & ok_lost3).astype(np.float32)
        w3_valid  = np.isfinite(W3_raw).astype(np.float32)
        dir_valid = np.isfinite(Dir3_raw).astype(np.float32)
        nac_valid = np.isfinite(Nac3_raw).astype(np.float32)

        P3 = np.nan_to_num(P3_raw, nan=0.0).astype(np.float32)
        W3 = np.nan_to_num(W3_raw, nan=0.0).astype(np.float32)

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
        Xv = np.zeros_like(X_scada, dtype=np.float32)
        Xv[..., 0] = p3_valid
        Xv[..., 1] = dp_valid
        Xv[..., 2] = w3_valid
        Xv[..., 3] = dir_valid
        Xv[..., 4] = dir_valid
        Xv[..., 5] = nac_valid
        Xv[..., 6] = nac_valid

        X_list.append(X_scada)
        Xv_list.append(Xv)

        # --- CERRA dynamic ---
        C, Cv, names = read_cerra_for_turbine_all_years(tid, full_time_x)

        # --- Optimization-3: append neighbors coords (dx/dy/dist) constant ---
        coord_feat, coord_names = load_neighbors_coords(tid)  # (3K,)
        T3 = C.shape[0]
        coord_rep = np.repeat(coord_feat[None, :], T3, axis=0).astype(np.float32)  # (T3,3K)
        coord_valid = np.ones_like(coord_rep, dtype=np.float32)

        C = np.concatenate([C, coord_rep], axis=-1).astype(np.float32)
        Cv = np.concatenate([Cv, coord_valid], axis=-1).astype(np.float32)
        names = names + coord_names

        if cerra_feature_names is None:
            cerra_feature_names = names
        else:
            if len(names) != len(cerra_feature_names):
                raise RuntimeError(f"CERRA feature dim mismatch: got {len(names)} vs {len(cerra_feature_names)}")

        C_feat_list.append(C)
        C_valid_list.append(Cv)

    X_scada = np.stack(X_list, axis=1).astype(np.float32)      # (T3,N,7)
    Xv_scada = np.stack(Xv_list, axis=1).astype(np.float32)    # (T3,N,7)

    C_feat = np.stack(C_feat_list, axis=1).astype(np.float32)  # (T3,N,Fc)
    C_valid = np.stack(C_valid_list, axis=1).astype(np.float32)

    X_raw = np.concatenate([X_scada, C_feat], axis=-1).astype(np.float32)
    X_valid = np.concatenate([Xv_scada, C_valid], axis=-1).astype(np.float32)

    Y_raw = np.stack(P1_list, axis=1).astype(np.float32)       # (T1,N)
    mask_y = np.stack(M1_list, axis=1).astype(np.float32)

    T3, N, F = X_raw.shape
    T1 = Y_raw.shape[0]

    # feature names
    feature_names = [
        "P3","dP3","W3","dir_sin3","dir_cos3","nac_sin3","nac_cos3"
    ] + (cerra_feature_names if cerra_feature_names else [])

    # detect coord feature indices => FORCE no z-score
    coord_mask = np.zeros((F,), dtype=bool)
    for i, nm in enumerate(feature_names):
        if nm.startswith("dx_k") or nm.startswith("dy_k") or nm.startswith("dist_k"):
            coord_mask[i] = True

    # -------- normalize only on train years & valid points ----------
    x_train_end = max(0, min(int(x_2019), T3))
    y_train_end = max(0, min(int(y_2019), T1))

    x_mu = np.zeros((F,), dtype=np.float32)
    x_sd = np.ones((F,), dtype=np.float32)

    Xtr = X_raw[:x_train_end]
    Vtr = X_valid[:x_train_end] > 0.5
    for f in range(F):
        if coord_mask[f]:
            x_mu[f] = 0.0
            x_sd[f] = 1.0
            continue
        v = Xtr[..., f]
        m = Vtr[..., f]
        if np.any(m):
            vals = v[m]
            mu = float(vals.mean())
            sd = float(vals.std())
            if sd < 1e-5:
                x_mu[f] = 0.0
                x_sd[f] = 1.0
            else:
                x_mu[f] = mu
                x_sd[f] = sd + 1e-6
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

    # -------- adj + turbine coords ----------
    static_path = os.path.join(RAW_DIR, "Kelmarsh_WT_static.csv")
    if os.path.exists(static_path):
        A, latlon, xy = build_adj_and_xy_from_static(static_path, turbine_ids)
    else:
        A = np.ones((N, N), dtype=np.float32) / N
        latlon = np.zeros((N,2), dtype=np.float32)
        xy = np.zeros((N,2), dtype=np.float32)

    # -------- save ----------
    np.save(f"{OUT_DIR}/X.npy", Xn.astype(np.float32))
    np.save(f"{OUT_DIR}/X_valid.npy", X_valid.astype(np.float32))
    np.save(f"{OUT_DIR}/Y.npy", Yn.astype(np.float32))
    np.save(f"{OUT_DIR}/mask.npy", mask_y.astype(np.float32))
    np.save(f"{OUT_DIR}/adj.npy", A.astype(np.float32))

    # -------- S2 meta mapping ----------
    scada_idx = list(range(7))
    cerra_idx0 = 7
    cerra_names = feature_names[cerra_idx0:]

    cerra_an_idx = []
    cerra_fc_idx_by_lead = {str(h): [] for h in FORECAST_LEADS}
    for i, nm in enumerate(cerra_names):
        gi = cerra_idx0 + i
        m = re.search(r"_fc(\d+)_75$", nm)
        if m:
            lead = m.group(1)
            if lead in cerra_fc_idx_by_lead:
                cerra_fc_idx_by_lead[lead].append(gi)
        else:
            cerra_an_idx.append(gi)

    meta = {
        "turbine_ids": turbine_ids,
        "turbine_latlon": latlon.tolist(),
        "turbine_xy": xy.tolist(),
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

        "x_start": str(t_x0),
        "x_end":   str(t_x1),
        "y_start": str(t_y0),
        "y_end":   str(t_y1),

        "splits": {
            "train_x": [0, int(x_2019)],
            "val_x":   [int(x_2019), int(x_2020)],
            "test_x":  [int(x_2020), int(T3)],
            "train_y": [0, int(y_2019)],
            "val_y":   [int(y_2019), int(y_2020)],
            "test_y":  [int(y_2020), int(T1)],
        },

        "s2": {
            "scada_idx": scada_idx,
            "cerra_an_idx": cerra_an_idx,
            "cerra_fc_idx_by_lead": cerra_fc_idx_by_lead,
            "forecast_leads": FORECAST_LEADS,

            "k_neighbors": int(K_NEIGHBORS),
            "coord_feat": ["dx", "dy", "dist"],

            # IMPORTANT: declare dynamic column layout for bg encoder
            # Your CSV ordering is: an(all K points) then fc1(all K points) ... fc6(all K points)
            "bg_dyn_layout": "block_major",
            "bg_feat_per_point": 4,
            "bg_blocks": ["an"] + [f"fc{h}" for h in FORECAST_LEADS],
            "bg_point_feat_order": ["speed", "direction", "u", "v"],
        }
    }

    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[OK] Saved to", OUT_DIR)
    print("X", Xn.shape, "X_valid", X_valid.shape, "Y", Yn.shape, "mask", mask_y.shape, "adj", A.shape)
    print("Features:", len(feature_names))
    print("S2: cerra_an =", len(cerra_an_idx), "cerra_fc (per lead) =", {k: len(v) for k,v in cerra_fc_idx_by_lead.items()})
    print("Coord features appended =", int(coord_mask.sum()))

if __name__ == "__main__":
    main()