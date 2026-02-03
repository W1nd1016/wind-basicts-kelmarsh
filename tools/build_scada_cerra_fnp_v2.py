# tools/build_scada_cerra_fnp_v2.py
import os, re, json, glob
import numpy as np
import pandas as pd

# =========================
# Config
# =========================
RAW_DIR   = "raw_scada"
CERRA_DIR = "raw_new_cerra1hour"
OUT_DIR   = "data/wind_scada_cerra_fnp_v2"
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

# =========================
# Utils
# =========================
def turbine_id_from_filename(fname: str):
    m = re.search(r"Kelmarsh_(\d+)_", fname)
    return int(m.group(1)) if m else None

def read_one_scada_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=9)
    df.columns = [c.strip() for c in df.columns]
    need = {TIME_COL, PWR_COL, WSP_COL, WDIR_COL, NAC_COL, LOST_COL}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"{os.path.basename(path)} missing columns: {sorted(miss)}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL)
    return df[[TIME_COL, PWR_COL, WSP_COL, WDIR_COL, NAC_COL, LOST_COL]]

def is_ok_lost(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    ok = finite & (arr == 0)
    return ok.astype(np.float32)

def idx_at(ts: pd.Timestamp, base: pd.Timestamp, step_h: int) -> int:
    return int((ts - base) / pd.Timedelta(hours=step_h))

def parse_year_from_cerra_fname(fname: str):
    # expects ..._YYYY.csv
    m = re.search(r"_(\d{4})\.csv$", fname)
    return int(m.group(1)) if m else None

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

def load_neighbors_npz(tid: int):
    """
    read raw_new_cerra1hour/CERRA_75m_KWF{tid}_neighbors.npz
    returns dx_m, dy_m, dist_m (K,)
    """
    npz_path = os.path.join(CERRA_DIR, f"CERRA_75m_KWF{tid}_neighbors.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing neighbors npz: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    dx_m = z["dx_m"].astype(np.float32)
    dy_m = z["dy_m"].astype(np.float32)
    dist_m = z["dist_m"].astype(np.float32)
    if dx_m.shape[0] != K_NEIGHBORS or dy_m.shape[0] != K_NEIGHBORS or dist_m.shape[0] != K_NEIGHBORS:
        raise RuntimeError(f"Neighbor K mismatch in {npz_path}: got {dx_m.shape}, {dy_m.shape}, {dist_m.shape}, expect K={K_NEIGHBORS}")
    return dx_m, dy_m, dist_m

def read_cerra_uv_for_turbine_all_years(tid: int, full_time_x: pd.DatetimeIndex):
    """
    Read all CERRA yearly CSVs for turbine KWF{tid}, align to full_time_x (3h).
    Only keep u/v columns (analysis + fc1..fc6), drop speed/direction.

    Return:
      C:   (T3, Fbg)
      Cv:  (T3, Fbg) 0/1
      names: list[str] with _75 suffix
      bg_maps: dict with indices grouped by (an, fc lead)
    """
    tur_name = f"KWF{tid}"
    pattern_year = os.path.join(CERRA_DIR, f"CERRA_75m_{tur_name}_*.csv")
    paths_year = sorted(glob.glob(pattern_year))
    if not paths_year:
        raise FileNotFoundError(f"Cannot find CERRA yearly files: {pattern_year}")

    dfs = []
    for p in paths_year:
        y = parse_year_from_cerra_fname(os.path.basename(p))
        if y is None or y not in ALL_YEARS:
            continue
        df = pd.read_csv(p)
        if "time" not in df.columns:
            raise KeyError(f"{p} missing 'time' column")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"{tur_name} has no CERRA data for years {sorted(ALL_YEARS)}")

    df_all = pd.concat(dfs, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"]).set_index("time")

    # --- build desired column order (block-major by lead, inside block by k, variables u/v) ---
    want_cols = []
    # analysis @ base time
    for k in range(1, K_NEIGHBORS + 1):
        want_cols.append(f"u_k{k}")
        want_cols.append(f"v_k{k}")
    # forecast @ base time (storage A): u_k{}/v_k{}_{fcLead}
    for lead in FORECAST_LEADS:
        for k in range(1, K_NEIGHBORS + 1):
            want_cols.append(f"u_k{k}_fc{lead}")
            want_cols.append(f"v_k{k}_fc{lead}")

    missing = [c for c in want_cols if c not in df_all.columns]
    if missing:
        # show a small helpful subset
        show = missing[:20]
        raise KeyError(
            f"{tur_name} CERRA missing {len(missing)} required u/v columns. "
            f"First missing: {show}. "
            f"Existing example cols: {list(df_all.columns)[:30]}"
        )

    sub = df_all[want_cols].copy()

    # align to full_time_x (3h), keep NaN for missing times
    sub = sub.reindex(full_time_x)

    valid = np.isfinite(sub.values).astype(np.float32)
    sub = sub.fillna(0.0)

    names = [f"{c}_75" for c in want_cols]
    C = sub.values.astype(np.float32)

    # build maps (indices within background feature block)
    # background features are exactly in this order
    an_idx = []
    fc_idx_by_lead = {str(h): [] for h in FORECAST_LEADS}
    for i, c in enumerate(want_cols):
        m = re.search(r"_fc(\d+)$", c)
        if m:
            lead = m.group(1)
            fc_idx_by_lead[lead].append(i)
        else:
            an_idx.append(i)

    bg_maps = {
        "layout": "block_major",
        "k_neighbors": int(K_NEIGHBORS),
        "vars_per_point": 2,  # u,v
        "blocks": ["an"] + [f"fc{h}" for h in FORECAST_LEADS],
        "an_idx": an_idx,
        "fc_idx_by_lead": fc_idx_by_lead,
        "col_order_example": {
            "an": ["u_k1","v_k1","u_k2","v_k2", "..."],
            "fc1": ["u_k1_fc1","v_k1_fc1","u_k2_fc1","v_k2_fc1", "..."]
        }
    }

    return C, valid, names, bg_maps

# =========================
# Main
# =========================
def main():
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "Turbine_Data_Kelmarsh_*.csv")))
    if not all_files:
        raise FileNotFoundError(f"No Turbine_Data_Kelmarsh_*.csv in {RAW_DIR}")

    by_tid = {}
    for f in all_files:
        tid = turbine_id_from_filename(os.path.basename(f))
        if tid is None:
            continue
        by_tid.setdefault(tid, []).append(f)

    turbine_ids = sorted(by_tid.keys())
    if len(turbine_ids) != 6:
        raise RuntimeError(f"Expect 6 turbines, got {len(turbine_ids)}: {turbine_ids}")

    # ---- read all scada per turbine ----
    turb_raw = {}
    for tid in turbine_ids:
        dfs = []
        for f in sorted(by_tid[tid]):
            dfs.append(read_one_scada_file(f))
        df_all = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=[TIME_COL]).sort_values(TIME_COL)
        df_all = df_all[df_all[TIME_COL].dt.year.isin(ALL_YEARS)].copy()
        turb_raw[tid] = df_all

    # ---- time axis ----
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
    if ratio != 3 or Y_STEP_HOURS != 1 or X_STEP_HOURS != 3:
        raise RuntimeError("This v2 builder assumes X_STEP_HOURS=3 and Y_STEP_HOURS=1 (ratio=3).")

    # split boundaries (keep same as v1)
    x_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    x_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    y_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_y0, Y_STEP_HOURS)
    y_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_y0, Y_STEP_HOURS)

    # ---- build arrays ----
    P1_list, M1_list = [], []
    X_list, Xv_list = [], []
    C_list, Cv_list = [], []
    cerra_feature_names = None
    bg_maps_any = None

    # neighbors
    dx_all, dy_all, dist_all = [], [], []

    for tid in turbine_ids:
        df = turb_raw[tid]
        dft = df.set_index(TIME_COL).sort_index()

        # hourly SCADA core
        sub_hour = dft[[PWR_COL, WSP_COL, WDIR_COL, NAC_COL]].copy()
        sub_hour = sub_hour[(sub_hour.index.minute == 0) & (sub_hour.index.second == 0)]
        sub_hour = sub_hour.groupby(level=0).mean(numeric_only=True)
        sub_hour = sub_hour.reindex(full_time_y)

        # hourly downtime/curtailment (sum)
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

        # build 3h inputs
        W1_raw   = sub_hour[WSP_COL].to_numpy(dtype=np.float32)
        Dir1_raw = sub_hour[WDIR_COL].to_numpy(dtype=np.float32)
        Nac1_raw = sub_hour[NAC_COL].to_numpy(dtype=np.float32)

        P3_raw   = pd.Series(P1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        W3_raw   = pd.Series(W1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Dir3_raw = pd.Series(Dir1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Nac3_raw = pd.Series(Nac1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)

        # 3h lost: strict (must have all 3 hours finite)
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

        # angles -> sin/cos (stable)
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

        # delta power
        dP = np.diff(P3, axis=0, prepend=P3[[0]])
        prev = np.roll(p3_valid, 1); prev[0] = 0.0
        dp_valid = ((p3_valid > 0.5) & (prev > 0.5)).astype(np.float32)
        dP[dp_valid < 0.5] = 0.0

        P3[p3_valid < 0.5] = 0.0
        W3[w3_valid < 0.5] = 0.0

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

        # CERRA background u/v only
        C, Cv, names, bg_maps = read_cerra_uv_for_turbine_all_years(tid, full_time_x)

        if cerra_feature_names is None:
            cerra_feature_names = names
            bg_maps_any = bg_maps
        else:
            if len(names) != len(cerra_feature_names):
                raise RuntimeError(f"CERRA feature dim mismatch: got {len(names)} vs {len(cerra_feature_names)}")

        C_list.append(C.astype(np.float32))
        Cv_list.append(Cv.astype(np.float32))

        # neighbors raw meters
        dx_m, dy_m, dist_m = load_neighbors_npz(tid)
        dx_all.append(dx_m)
        dy_all.append(dy_m)
        dist_all.append(dist_m)

    # stack shapes
    X_scada = np.stack(X_list, axis=1).astype(np.float32)      # (T3,N,7)
    Xv_scada = np.stack(Xv_list, axis=1).astype(np.float32)    # (T3,N,7)
    C_feat = np.stack(C_list, axis=1).astype(np.float32)       # (T3,N,Fbg)
    C_valid = np.stack(Cv_list, axis=1).astype(np.float32)

    # final X
    X_raw = np.concatenate([X_scada, C_feat], axis=-1).astype(np.float32)    # (T3,N,F)
    X_valid = np.concatenate([Xv_scada, C_valid], axis=-1).astype(np.float32)

    # hourly Y raw (T1,N)
    Y_raw = np.stack(P1_list, axis=1).astype(np.float32)
    M_raw = np.stack(M1_list, axis=1).astype(np.float32)

    T3, N, F = X_raw.shape
    T1 = Y_raw.shape[0]

    # ---- build Y sequence aligned to each x_idx (base time at full_time_x) ----
    # y_idx = offset + x_idx*ratio, target uses (y_idx+1 ... y_idx+H) => future 1..H hours
    Y_seq = np.zeros((T3, H, N), dtype=np.float32)
    Yv_seq = np.zeros((T3, H, N), dtype=np.float32)

    for x_idx in range(T3):
        y_idx = offset_hours + x_idx * ratio
        for h in range(1, H+1):
            yi = y_idx + h
            if 0 <= yi < T1:
                Y_seq[x_idx, h-1, :] = Y_raw[yi, :]
                Yv_seq[x_idx, h-1, :] = M_raw[yi, :]
            else:
                # out of range -> keep zero and invalid
                pass

    # ---- feature names ----
    feature_names = [
        "P3","dP3","W3","dir_sin3","dir_cos3","nac_sin3","nac_cos3"
    ] + (cerra_feature_names if cerra_feature_names else [])

    # ---- normalization (train only, valid only) ----
    x_train_end = max(0, min(int(x_2019), T3))
    y_train_end = max(0, min(int(y_2019), T1))

    x_mu = np.zeros((F,), dtype=np.float32)
    x_sd = np.ones((F,), dtype=np.float32)

    Xtr = X_raw[:x_train_end]
    Vtr = X_valid[:x_train_end] > 0.5
    for f_i in range(F):
        v = Xtr[..., f_i]
        m = Vtr[..., f_i]
        if np.any(m):
            vals = v[m]
            mu = float(vals.mean())
            sd = float(vals.std())
            if sd < 1e-5:
                x_mu[f_i] = 0.0
                x_sd[f_i] = 1.0
            else:
                x_mu[f_i] = mu
                x_sd[f_i] = sd + 1e-6
        else:
            x_mu[f_i] = 0.0
            x_sd[f_i] = 1.0

    Ytr = Y_raw[:y_train_end]
    Mtr = M_raw[:y_train_end] > 0.5
    if np.any(Mtr):
        y_mu = float(Ytr[Mtr].mean())
        y_sd = float(Ytr[Mtr].std()) + 1e-6
    else:
        y_mu, y_sd = 0.0, 1.0

    Xn = (X_raw - x_mu) / x_sd
    Xn[X_valid < 0.5] = 0.0

    Yn = (Y_seq - y_mu) / y_sd
    Yn[Yv_seq < 0.5] = 0.0

    # ---- adjacency + turbine coords ----
    static_path = os.path.join(RAW_DIR, "Kelmarsh_WT_static.csv")
    if os.path.exists(static_path):
        A, latlon, xy = build_adj_and_xy_from_static(static_path, turbine_ids)
    else:
        A = np.ones((N, N), dtype=np.float32) / N
        latlon = np.zeros((N,2), dtype=np.float32)
        xy = np.zeros((N,2), dtype=np.float32)

    # ---- neighbors: raw meters + global scale for later normalization in model ----
    dx_m_mat = np.stack(dx_all, axis=0).astype(np.float32)       # (N,K)
    dy_m_mat = np.stack(dy_all, axis=0).astype(np.float32)
    dist_m_mat = np.stack(dist_all, axis=0).astype(np.float32)

    dx_scale = float(np.max(np.abs(dx_m_mat)) + 1e-6)
    dy_scale = float(np.max(np.abs(dy_m_mat)) + 1e-6)
    dist_scale = float(np.max(dist_m_mat) + 1e-6)

    # ---- save arrays ----
    np.save(f"{OUT_DIR}/X.npy", Xn.astype(np.float32))                 # (T3,N,F)
    np.save(f"{OUT_DIR}/X_valid.npy", X_valid.astype(np.float32))     # (T3,N,F)
    np.save(f"{OUT_DIR}/Y.npy", Yn.astype(np.float32))                 # (T3,H,N)
    np.save(f"{OUT_DIR}/Y_valid.npy", Yv_seq.astype(np.float32))       # (T3,H,N)
    np.save(f"{OUT_DIR}/adj.npy", A.astype(np.float32))

    np.save(f"{OUT_DIR}/neighbors_dx_m.npy", dx_m_mat)
    np.save(f"{OUT_DIR}/neighbors_dy_m.npy", dy_m_mat)
    np.save(f"{OUT_DIR}/neighbors_dist_m.npy", dist_m_mat)

    # ---- meta ----
    meta = {
        "turbine_ids": turbine_ids,
        "turbine_latlon": latlon.tolist(),
        "turbine_xy": xy.tolist(),

        "feature_names": feature_names,
        "scada_dim": 7,
        "bg_dim": int(len(feature_names) - 7),

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

        "neighbors": {
            "k_neighbors": int(K_NEIGHBORS),
            "dx_scale": dx_scale,
            "dy_scale": dy_scale,
            "dist_scale": dist_scale,
            "units": "meters",
            "files": {
                "dx_m": "neighbors_dx_m.npy",
                "dy_m": "neighbors_dy_m.npy",
                "dist_m": "neighbors_dist_m.npy"
            }
        },

        # background mapping inside bg block (before concatenating with scada)
        "bg_maps": bg_maps_any
    }

    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[OK] Saved to", OUT_DIR)
    print("X", Xn.shape, "X_valid", X_valid.shape)
    print("Y", Yn.shape, "Y_valid", Yv_seq.shape)
    print("adj", A.shape)
    print("neighbors", dx_m_mat.shape, dy_m_mat.shape, dist_m_mat.shape)
    print("Features:", len(feature_names), "scada=7 bg=", len(feature_names)-7)
    print("bg_maps:", {k: (len(v) if isinstance(v, list) else "dict") for k,v in (bg_maps_any or {}).items()})

if __name__ == "__main__":
    main()
