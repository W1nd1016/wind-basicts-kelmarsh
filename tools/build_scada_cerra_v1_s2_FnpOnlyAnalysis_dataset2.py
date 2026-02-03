# tools/build_scada_cerra_v1_s2_FnpOnlyAnalysis_dataset2.py
import os, re, json, glob
import numpy as np
import pandas as pd

# =========================
# Dataset2 (Penmanshiel) paths
# =========================
RAW_DIR   = "raw_scada_dataset2"
CERRA_DIR = "raw_new_cerra1hour_dataset2"
OUT_DIR   = "data/wind_scada_cerra_v1_s2_FnpOnlyAnalysis_dataset2"
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

# 变量顺序固定：每个点 (K) 的 4 个变量
CERRA_VAR_ORDER = ["speed", "direction", "u", "v"]


# -------------------------
# Helpers: turbine id/tag parsing for Dataset2
# -------------------------
def turbine_id_from_filename(fname: str):
    """
    Dataset2 SCADA filename example:
      Turbine_Data_Penmanshiel_01_2016-06-06_-_2017-01-01_1042.csv
    Return int turbine id, e.g., 1 for "_01_".
    """
    m = re.search(r"Penmanshiel_(\d+)_", fname)
    if m:
        return int(m.group(1))
    # fallback: find _Txx_ or _xx_ if naming varies
    m2 = re.search(r"_T(\d+)_", fname)
    if m2:
        return int(m2.group(1))
    m3 = re.search(r"_(\d{1,2})_", fname)
    return int(m3.group(1)) if m3 else None


def tid_to_cerra_tag(tid: int) -> str:
    """
    Map integer turbine id to the tag used in dataset2 CERRA filenames.

    Examples:
      tid=1  -> T01   (CERRA_75m_T01_2016.csv)
      tid=10 -> T010  (CERRA_75m_T010_2016.csv)

    Note: dataset2 uses 2-digit tags for 1..9 and 3-digit tags for 10..15.
    """
    tid = int(tid)
    if tid >= 10:
        return f"T{tid:03d}"
    return f"T{tid:02d}"

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


def _pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def build_adj_and_xy_from_static(static_path, turbine_ids):
    """
    Robust static reader:
    - Accepts different id columns / name columns.
    - Drops rows where turbine id cannot be parsed.
    - Requires all turbine_ids to be present after filtering (otherwise raise).
    """
    sdf = pd.read_csv(static_path)
    sdf.columns = [c.strip() for c in sdf.columns]

    # ---- turbine id parsing ----
    if "turbine_id" in sdf.columns:
        tid_ser = pd.to_numeric(sdf["turbine_id"], errors="coerce")
    else:
        # common candidates for name/id columns
        id_col = _pick_col(
            sdf.columns,
            ["Turbine ID", "TurbineID", "ID", "Id", "id", "WT", "WT_ID", "wt_id"]
        )
        name_col = _pick_col(
            sdf.columns,
            ["Title", "Alternative Title", "Name", "Turbine", "WT Name", "WT_Name"]
        )

        if id_col is not None:
            tid_ser = pd.to_numeric(sdf[id_col], errors="coerce")
        elif name_col is not None:
            # extract digits from strings like "T01", "Penmanshiel 01", etc.
            extracted = sdf[name_col].astype(str).str.extract(r"(\d+)")
            tid_ser = pd.to_numeric(extracted[0], errors="coerce")
        else:
            raise KeyError(
                f"Static file {os.path.basename(static_path)} 缺少可识别的 turbine id 列。"
                f"现有列: {list(sdf.columns)}"
            )

    sdf = sdf.copy()
    sdf["turbine_id"] = tid_ser
    sdf = sdf.dropna(subset=["turbine_id"])
    sdf["turbine_id"] = sdf["turbine_id"].astype(int)

    # ---- lat/lon parsing ----
    lat_col = _pick_col(sdf.columns, ["Latitude", "lat", "Lat", "LAT"])
    lon_col = _pick_col(sdf.columns, ["Longitude", "lon", "Lon", "LON", "Long", "LONG"])

    if lat_col is None or lon_col is None:
        raise KeyError(
            f"Static file {os.path.basename(static_path)} 缺少 Latitude/Longitude 列。"
            f"现有列: {list(sdf.columns)}"
        )

    sdf = sdf.rename(columns={lat_col: "lat", lon_col: "lon"})

    # keep only requested turbines
    sdf = sdf[sdf["turbine_id"].isin(turbine_ids)].copy()
    sdf = sdf.drop_duplicates(subset=["turbine_id"]).sort_values("turbine_id")

    # require complete coverage (avoid silently mismatching node order)
    got = set(sdf["turbine_id"].tolist())
    need = set(map(int, turbine_ids))
    if got != need:
        missing = sorted(list(need - got))
        extra = sorted(list(got - need))
        raise RuntimeError(
            f"Static file 覆盖不完整：missing={missing}, extra={extra}. "
            f"请检查 {static_path} 的 turbine id/名称列是否包含所有风机。"
        )

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

    sigma = np.median(d[d>0]) if np.any(d[d>0]) else 1.0
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


def _cerra_year_paths(tag: str):
    # Dataset2: CERRA_75m_T01_2016.csv
    pattern = os.path.join(CERRA_DIR, f"CERRA_75m_{tag}_*.csv")
    return sorted(glob.glob(pattern))


def load_neighbors_pos(tag: str):
    """
    读取 raw_new_cerra1hour_dataset2/CERRA_75m_{tag}_neighbors.npz
    返回:
      pos_norm: (K,3)  = (dx,dy,dist) 均已按各自 max 归一化到 ~[-1,1]/[0,1]
      scales: dict, 方便写入 meta
    """
    npz_path = os.path.join(CERRA_DIR, f"CERRA_75m_{tag}_neighbors.npz")
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

    pos = np.stack([dx, dy, ds], axis=-1).astype(np.float32)  # (K,3)
    scales = {"dx_scale": dx_scale, "dy_scale": dy_scale, "dist_scale": dist_scale}
    return pos, scales


def read_cerra_analysis_all_years(tag: str, full_time_x: pd.DatetimeIndex):
    """
    读取 CERRA csv（多年份），只取 analysis 列（不含 _fc）
    输出对齐到 full_time_x（3h）
    返回:
      an_vals : (T3, K*4)
      an_valid: (T3, K*4)
      an_names: list[str]  每列已加 _75 后缀
    """
    paths_year = _cerra_year_paths(tag)
    if not paths_year:
        raise FileNotFoundError(f"找不到 CERRA 年文件: {os.path.join(CERRA_DIR, f'CERRA_75m_{tag}_*.csv')}")

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
        raise FileNotFoundError(f"{tag} 没有任何 {sorted(ALL_YEARS)} 年份的 CERRA 数据")

    df_all = pd.concat(dfs, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
    df_all = df_all.reindex(full_time_x)

    # analysis 列：形如 speed_k1, direction_k1, u_k1, v_k1 ... (不含 _fc)
    cols = []
    for k in range(1, K_NEIGHBORS + 1):
        for v in CERRA_VAR_ORDER:
            cols.append(f"{v}_k{k}")

    miss = [c for c in cols if c not in df_all.columns]
    if miss:
        raise KeyError(f"[{tag}] 缺少 analysis 列: {miss[:6]} ... total_missing={len(miss)}")

    sub = df_all[cols].copy()
    valid = np.isfinite(sub.values).astype(np.float32)
    sub = sub.fillna(0.0)

    names = [f"{c}_75" for c in cols]
    return sub.values.astype(np.float32), valid.astype(np.float32), names


def read_cerra_forecast_all_years(tag: str, full_time_x: pd.DatetimeIndex):
    """
    读取 CERRA csv（多年份），只取 forecast@t0（lead=1..H），并保留 16 个格点全部变量
    输出对齐到 full_time_x（3h）

    返回:
      fc_vals : (T3, H, K, 4)   4 顺序 = [speed, direction, u, v]
      fc_valid: (T3, H, K, 4)
    """
    paths_year = _cerra_year_paths(tag)
    if not paths_year:
        raise FileNotFoundError(f"找不到 CERRA 年文件: {os.path.join(CERRA_DIR, f'CERRA_75m_{tag}_*.csv')}")

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
        raise FileNotFoundError(f"{tag} 没有任何 {sorted(ALL_YEARS)} 年份的 CERRA 数据")

    df_all = pd.concat(dfs, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
    df_all = df_all.reindex(full_time_x)

    T3 = len(full_time_x)
    fc = np.zeros((T3, H, K_NEIGHBORS, 4), dtype=np.float32)
    fv = np.zeros((T3, H, K_NEIGHBORS, 4), dtype=np.float32)

    # lead-major, then k, then var-order
    for li, lead in enumerate(FORECAST_LEADS[:H]):
        for k in range(1, K_NEIGHBORS + 1):
            for vi, vname in enumerate(CERRA_VAR_ORDER):
                col = f"{vname}_k{k}_fc{lead}"
                if col not in df_all.columns:
                    raise KeyError(f"[{tag}] 缺少 forecast 列: {col}")
                arr = df_all[col].to_numpy(dtype=np.float32)
                ok = np.isfinite(arr).astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
                fc[:, li, k-1, vi] = arr
                fv[:, li, k-1, vi] = ok

    return fc.astype(np.float32), fv.astype(np.float32)


def idx_at(ts: pd.Timestamp, base: pd.Timestamp, step_h: int) -> int:
    return int((ts - base) / pd.Timedelta(hours=step_h))


def _auto_find_static_file(raw_dir: str):
    # Prefer dataset2 naming, then any *_WT_static.csv, then any *_static.csv
    cands = [
        os.path.join(raw_dir, "Penmanshiel_WT_static.csv"),
        os.path.join(raw_dir, "Kelmarsh_WT_static.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p

    c2 = sorted(glob.glob(os.path.join(raw_dir, "*_WT_static.csv")))
    if len(c2) == 1:
        return c2[0]
    if len(c2) > 1:
        raise RuntimeError(f"RAW_DIR 下发现多个 *_WT_static.csv，请手动指定: {c2}")

    c3 = sorted(glob.glob(os.path.join(raw_dir, "*_static.csv")))
    if len(c3) == 1:
        return c3[0]
    if len(c3) > 1:
        raise RuntimeError(f"RAW_DIR 下发现多个 *_static.csv，请手动指定: {c3}")

    return None


def main():
    # Penmanshiel SCADA pattern
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "Turbine_Data_Penmanshiel_*.csv")))
    if not all_files:
        raise FileNotFoundError(f"{RAW_DIR} 下找不到 Turbine_Data_Penmanshiel_*.csv")

    by_tid = {}
    for f in all_files:
        tid = turbine_id_from_filename(os.path.basename(f))
        if tid is None:
            continue
        by_tid.setdefault(tid, []).append(f)

    turbine_ids = sorted(by_tid.keys())
    if len(turbine_ids) < 2:
        raise RuntimeError(f"解析到的风机数量太少: {turbine_ids}")

    # 读取所有 SCADA
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

    # split boundaries (by absolute calendar years)
    x_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    x_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    y_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_y0, Y_STEP_HOURS)
    y_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_y0, Y_STEP_HOURS)

    # -------- build arrays ----------
    P1_list, M1_list = [], []
    X_list, Xv_list = [], []
    AN_list, ANv_list = [], []
    FC_list, FCv_list = [], []
    pos_list = []

    cerra_an_feature_names = None

    # To keep consistent node order, we also store the CERRA tag order used.
    turbine_tags = [tid_to_cerra_tag(tid) for tid in turbine_ids]

    for tid, tag in zip(turbine_ids, turbine_tags):
        df = turb_raw[tid]
        dft = df.set_index(TIME_COL).sort_index()

        # --- hourly SCADA -> full_time_y ---
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

        # --- 3-hour aligned SCADA (for X) ---
        W1_raw   = sub_hour[WSP_COL].to_numpy(dtype=np.float32)
        Dir1_raw = sub_hour[WDIR_COL].to_numpy(dtype=np.float32)
        Nac1_raw = sub_hour[NAC_COL].to_numpy(dtype=np.float32)

        P3_raw   = pd.Series(P1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        W3_raw   = pd.Series(W1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Dir3_raw = pd.Series(Dir1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Nac3_raw = pd.Series(Nac1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)

        # lost3 (rolling 3 hours) -> aligned to full_time_x
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

        # --- CERRA analysis (history) ---
        AN, ANv, an_names = read_cerra_analysis_all_years(tag, full_time_x)  # (T3, K*4)
        if cerra_an_feature_names is None:
            cerra_an_feature_names = an_names
        else:
            if len(an_names) != len(cerra_an_feature_names):
                raise RuntimeError(f"CERRA analysis dim mismatch: got {len(an_names)} vs {len(cerra_an_feature_names)}")

        AN_list.append(AN)
        ANv_list.append(ANv)

        # --- CERRA forecast@t0 (future 6 hours) ---
        FC, FCv = read_cerra_forecast_all_years(tag, full_time_x)  # (T3,H,K,4)
        FC_list.append(FC)
        FCv_list.append(FCv)

        # --- neighbors pos (dx,dy,dist) normalized ---
        pos, scales = load_neighbors_pos(tag)  # (K,3)
        pos_list.append(pos)

    # stack SCADA
    X_scada = np.stack(X_list, axis=1).astype(np.float32)      # (T3,N,7)
    Xv_scada = np.stack(Xv_list, axis=1).astype(np.float32)    # (T3,N,7)

    # stack analysis
    AN_feat = np.stack(AN_list, axis=1).astype(np.float32)     # (T3,N,K*4)
    AN_valid = np.stack(ANv_list, axis=1).astype(np.float32)   # (T3,N,K*4)

    # stack forecast -> we want (T3,H,N,K,4)
    FC_feat = np.stack(FC_list, axis=2).astype(np.float32)     # (T3,H,N,K,4)
    FC_valid = np.stack(FCv_list, axis=2).astype(np.float32)   # (T3,H,N,K,4)

    # pos: (N,K,3)
    pos_all = np.stack(pos_list, axis=0).astype(np.float32)

    # merge X (no pos duplicated in X; pos 单独存)
    X_raw = np.concatenate([X_scada, AN_feat], axis=-1).astype(np.float32)      # (T3,N,7+K*4)
    X_valid = np.concatenate([Xv_scada, AN_valid], axis=-1).astype(np.float32) # (T3,N,7+K*4)

    # Y/hourly
    Y_raw = np.stack(P1_list, axis=1).astype(np.float32)       # (T1,N)
    mask_y = np.stack(M1_list, axis=1).astype(np.float32)

    T3, N, F = X_raw.shape
    T1 = Y_raw.shape[0]

    # feature names for X
    feature_names = [
        "P3","dP3","W3","dir_sin3","dir_cos3","nac_sin3","nac_cos3"
    ] + (cerra_an_feature_names if cerra_an_feature_names else [])

    # -------- normalize only on train years & valid points ----------
    x_train_end = max(0, min(int(x_2019), T3))
    y_train_end = max(0, min(int(y_2019), T1))

    x_mu = np.zeros((F,), dtype=np.float32)
    x_sd = np.ones((F,), dtype=np.float32)

    Xtr = X_raw[:x_train_end]
    Vtr = X_valid[:x_train_end] > 0.5
    for f in range(F):
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

    # -------- normalize forecast FC on train_x & valid (separately from X) ----------
    # FC_feat: (T3,H,N,K,4)
    FCtr = FC_feat[:x_train_end]
    FVtr = (FC_valid[:x_train_end] > 0.5).astype(np.float32)

    eps = 1e-6
    cnt = FVtr.sum(axis=(0, 2)) + eps                # (H,K,4)
    s1 = (FCtr * FVtr).sum(axis=(0, 2))              # (H,K,4)
    s2 = ((FCtr * FVtr) ** 2).sum(axis=(0, 2))       # (H,K,4)

    fc_mu = (s1 / cnt).astype(np.float32)
    var = (s2 / cnt) - (fc_mu ** 2)
    var = np.maximum(var, 0.0).astype(np.float32)
    fc_sd = (np.sqrt(var) + 1e-6).astype(np.float32)

    # handle degenerate dims
    bad = (cnt <= 1.0) | (fc_sd < 1e-5)
    fc_mu[bad] = 0.0
    fc_sd[bad] = 1.0

    FCn = (FC_feat - fc_mu[None, :, None, :, :]) / fc_sd[None, :, None, :, :]
    FCn[FC_valid < 0.5] = 0.0

    # -------- adj + turbine coords ----------
    static_path = _auto_find_static_file(RAW_DIR)
    if static_path is not None and os.path.exists(static_path):
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

    # forecast + pos
    np.save(f"{OUT_DIR}/FC.npy", FCn.astype(np.float32))            # (T3,H,N,K,4)
    np.save(f"{OUT_DIR}/FC_valid.npy", FC_valid.astype(np.float32)) # (T3,H,N,K,4)
    np.save(f"{OUT_DIR}/pos.npy", pos_all.astype(np.float32))       # (N,K,3)

    meta = {
        "turbine_ids": turbine_ids,
        "turbine_tags": turbine_tags,

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

        "cerra": {
            "K": int(K_NEIGHBORS),
            "var_order": CERRA_VAR_ORDER,
            "analysis_dim_per_turbine": int(K_NEIGHBORS * 4),
            "analysis_layout": "k_major_then_var",
            "forecast_layout": "lead_major_then_k_then_var",
            "forecast_shape": [int(H), int(K_NEIGHBORS), 4],  # (H,K,4)
            "pos_shape": [int(K_NEIGHBORS), 3],               # (K,3)
            "pos_order": ["dx", "dy", "dist"],
            "fc_mu_shape": [int(H), int(K_NEIGHBORS), 4],
            "fc_sd_shape": [int(H), int(K_NEIGHBORS), 4],
        },

        "fc_mu": fc_mu.tolist(),
        "fc_sd": fc_sd.tolist(),
    }

    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[OK] Saved to", OUT_DIR)
    print("X", Xn.shape, "X_valid", X_valid.shape, "Y", Yn.shape, "mask", mask_y.shape, "adj", A.shape)
    print("FC", FCn.shape, "FC_valid", FC_valid.shape, "pos", pos_all.shape)
    print("Features:", len(feature_names), "CERRA_analysis_dim=", K_NEIGHBORS * 4)


if __name__ == "__main__":
    main()
