# tools/build_only_scada_5y_match_scada_cerra_rules_dataset3.py
import os, re, json, glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# Config (dataset3)
# =========================
RAW_DIR = "raw_scada_dataset3"
OUT_DIR = "data/wind_onlyscada_no_5b_dataset3"
META_FILE = "Hill_of_Towie_turbine_metadata.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# dataset3 file patterns (one per month)
TURBINE_PATTERN = "tblSCTurbine_*.csv"
GRID_PATTERN    = "tblSCTurGrid_*.csv"

# dataset3 raw columns
TIME_COL_RAW    = "TimeStamp"
STATION_COL_RAW = "StationId"

RAW_WSP_COL  = "wtc_AcWindSp_mean"
RAW_WDIR_SUB = "wtc_YawPos_mean"
RAW_NAC_COL  = "wtc_NacelPos_mean"
RAW_PWR_COL  = "wtc_ActPower_mean"

# canonical column names (match dataset1 script style)
TIME_COL = "TimeStamp"
PWR_COL  = "Power (kW)"
WSP_COL  = "Wind speed (m/s)"
WDIR_COL = "Wind direction (°)"        # 用 yawpos 替代
NAC_COL  = "Nacelle position (°)"      # 用 nacelpos

# ===== 多时间尺度设置（与dataset1一致）=====
X_STEP_HOURS = 3
Y_STEP_HOURS = 1
H = 6
L = 9

# ===== 年份划分（与dataset1一致）=====
TRAIN_YEARS = {2016, 2017, 2018}
VAL_YEARS   = {2019}
TEST_YEARS  = {2020}
ALL_YEARS   = TRAIN_YEARS | VAL_YEARS | TEST_YEARS


# =========================
# Helpers
# =========================
def _strip_col(c: str) -> str:
    return str(c).strip().lstrip("\ufeff")


def parse_year_month_from_fname(fname: str) -> Optional[Tuple[int, int]]:
    # tblSCTurbine_2016_01.csv / tblSCTurGrid_2016_01.csv
    m = re.search(r"_(\d{4})_(\d{2})\.csv$", fname)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def idx_at(ts: pd.Timestamp, base: pd.Timestamp, step_h: int) -> int:
    return int((ts - base) / pd.Timedelta(hours=step_h))


def turbine_sort_key(name: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", str(name))
    if m:
        return (int(m.group(1)), str(name))
    return (10**9, str(name))


def _to_datetime_mixed(s: pd.Series) -> pd.Series:
    """
    Robust datetime parser for mixed formats like:
      '2017-07-01 00:10:00' and '2017-07-01'
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed", cache=False)
    except TypeError:
        return pd.to_datetime(s, errors="coerce", cache=False)


def _read_header_cols(path: str) -> List[str]:
    df0 = pd.read_csv(path, nrows=0, low_memory=False)
    return [_strip_col(c) for c in df0.columns]


def _read_csv_by_cols(path: str, wanted_cols: List[str]) -> pd.DataFrame:
    want = set(wanted_cols)

    def usecols(c):
        return _strip_col(c) in want

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.columns = [_strip_col(c) for c in df.columns]

    miss = want - set(df.columns)
    if miss:
        raise KeyError(f"{os.path.basename(path)} missing columns: {sorted(miss)}")
    return df


def build_adj_and_xy_from_metadata(metadata_path: str, turbine_names: List[str]):
    """
    用 Hill_of_Towie_turbine_metadata 里的 Latitude/Longitude 构建邻接矩阵 A，
    并同时输出 latlon、xy_norm（给后续模型/coords用也方便）。
    """
    df = pd.read_csv(metadata_path)
    df.columns = [_strip_col(c) for c in df.columns]

    need = {"Turbine Name", "Station ID", "Latitude", "Longitude"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"[metadata] Missing columns: {sorted(miss)}; got={list(df.columns)[:30]} ...")

    df["Turbine Name"] = df["Turbine Name"].astype(str).str.strip()
    df["Station ID"] = pd.to_numeric(df["Station ID"], errors="coerce").astype("Int64")

    df = df[df["Turbine Name"].isin(turbine_names)].copy()
    if len(df) != len(turbine_names):
        got = set(df["Turbine Name"].tolist())
        exp = set(turbine_names)
        raise RuntimeError(f"[metadata] turbine mismatch: missing={sorted(exp-got)} extra={sorted(got-exp)}")

    df["__ord"] = df["Turbine Name"].apply(lambda x: turbine_names.index(x))
    df = df.sort_values("__ord")

    latlon = df[["Latitude", "Longitude"]].to_numpy(dtype=np.float32)  # (N,2) lat,lon
    station_ids = df["Station ID"].to_numpy(dtype=np.int64)

    N = latlon.shape[0]
    lat0 = float(latlon[:, 0].mean())
    scale_lat = 111000.0
    scale_lon = 111000.0 * np.cos(np.deg2rad(lat0))

    xy = np.stack([
        (latlon[:, 1] - latlon[:, 1].mean()) * scale_lon,
        (latlon[:, 0] - latlon[:, 0].mean()) * scale_lat
    ], axis=1).astype(np.float32)

    d = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            d[i, j] = np.linalg.norm(xy[i] - xy[j])

    sigma = np.median(d[d > 0]) if np.any(d[d > 0]) else 1.0
    A = np.exp(-(d**2) / (sigma**2)).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-6)

    xy_norm = xy.copy()
    s = np.std(xy_norm, axis=0, keepdims=True) + 1e-6
    xy_norm = (xy_norm / s).astype(np.float32)

    return A.astype(np.float32), latlon.astype(np.float32), xy_norm.astype(np.float32), station_ids


# =========================
# SCADA (monthly) load
# =========================
def read_one_month_scada(turbine_csv: Optional[str], grid_csv: Optional[str], station_keep: set,
                         verbose: bool = False) -> pd.DataFrame:
    """
    读取一个月份的两张表，合并成统一表：
      TimeStamp, StationId, Power(kW), WindSpeed, WindDir(yawpos), NacelPos(nacelpos)

    注意：
      - wind direction 用 wtc_YawPos_mean 替代
      - nacelle position 用 wtc_NacelPos_mean
      - 时间用 mixed parser
    """
    # ---- turbine side ----
    if turbine_csv is not None:
        dt = _read_csv_by_cols(
            turbine_csv,
            [TIME_COL_RAW, STATION_COL_RAW, RAW_WSP_COL, RAW_WDIR_SUB, RAW_NAC_COL]
        )
        dt = dt.rename(columns={
            TIME_COL_RAW: TIME_COL,
            STATION_COL_RAW: STATION_COL_RAW,
            RAW_WSP_COL: WSP_COL,
            RAW_WDIR_SUB: WDIR_COL,   # yawpos -> wind direction substitute
            RAW_NAC_COL: NAC_COL,
        })

        dt[TIME_COL] = _to_datetime_mixed(dt[TIME_COL])
        dt = dt.dropna(subset=[TIME_COL])

        dt[STATION_COL_RAW] = pd.to_numeric(dt[STATION_COL_RAW], errors="coerce").astype("Int64")
        dt = dt.dropna(subset=[STATION_COL_RAW])

        dt = dt[dt[STATION_COL_RAW].isin(list(station_keep))].copy()

        for c in [WSP_COL, WDIR_COL, NAC_COL]:
            dt[c] = pd.to_numeric(dt[c], errors="coerce")

        # 同一时刻重复 -> mean
        dt = dt.groupby([TIME_COL, STATION_COL_RAW], as_index=False).mean(numeric_only=True)

        if verbose:
            print(f"[SCADA turb] {os.path.basename(turbine_csv)} OK (WSP={RAW_WSP_COL}, WDIR=yaw={RAW_WDIR_SUB}, NAC={RAW_NAC_COL})")
    else:
        dt = pd.DataFrame(columns=[TIME_COL, STATION_COL_RAW, WSP_COL, WDIR_COL, NAC_COL])

    # ---- grid side (power) ----
    if grid_csv is not None:
        dg = _read_csv_by_cols(
            grid_csv,
            [TIME_COL_RAW, STATION_COL_RAW, RAW_PWR_COL]
        )
        dg = dg.rename(columns={
            TIME_COL_RAW: TIME_COL,
            STATION_COL_RAW: STATION_COL_RAW,
            RAW_PWR_COL: PWR_COL,
        })

        dg[TIME_COL] = _to_datetime_mixed(dg[TIME_COL])
        dg = dg.dropna(subset=[TIME_COL])

        dg[STATION_COL_RAW] = pd.to_numeric(dg[STATION_COL_RAW], errors="coerce").astype("Int64")
        dg = dg.dropna(subset=[STATION_COL_RAW])

        dg = dg[dg[STATION_COL_RAW].isin(list(station_keep))].copy()

        dg[PWR_COL] = pd.to_numeric(dg[PWR_COL], errors="coerce")
        dg = dg.groupby([TIME_COL, STATION_COL_RAW], as_index=False).mean(numeric_only=True)

        if verbose:
            print(f"[SCADA grid] {os.path.basename(grid_csv)} OK (PWR={RAW_PWR_COL})")
    else:
        dg = pd.DataFrame(columns=[TIME_COL, STATION_COL_RAW, PWR_COL])

    merged = pd.merge(dt, dg, on=[TIME_COL, STATION_COL_RAW], how="outer")
    merged = merged.sort_values([TIME_COL, STATION_COL_RAW])
    return merged


def load_scada_all_years(raw_dir: str, station_keep: set) -> Dict[int, pd.DataFrame]:
    turb_files = sorted(glob.glob(os.path.join(raw_dir, TURBINE_PATTERN)))
    grid_files = sorted(glob.glob(os.path.join(raw_dir, GRID_PATTERN)))

    turb_map: Dict[Tuple[int, int], str] = {}
    grid_map: Dict[Tuple[int, int], str] = {}

    for p in turb_files:
        ym = parse_year_month_from_fname(os.path.basename(p))
        if ym and ym[0] in ALL_YEARS:
            turb_map[ym] = p

    for p in grid_files:
        ym = parse_year_month_from_fname(os.path.basename(p))
        if ym and ym[0] in ALL_YEARS:
            grid_map[ym] = p

    keys = sorted(set(turb_map.keys()) | set(grid_map.keys()))
    if not keys:
        raise FileNotFoundError(f"No monthly SCADA files found under {raw_dir} for years={sorted(ALL_YEARS)}")

    by_station: Dict[int, List[pd.DataFrame]] = {int(s): [] for s in station_keep}

    for i, (y, m) in enumerate(keys):
        tpath = turb_map.get((y, m), None)
        gpath = grid_map.get((y, m), None)
        if tpath is None and gpath is None:
            continue

        verbose = (i < 3)  # 只打印前3个月，避免刷屏
        dfm = read_one_month_scada(tpath, gpath, station_keep=station_keep, verbose=verbose)
        if len(dfm) == 0:
            continue

        for sid in dfm[STATION_COL_RAW].dropna().unique():
            sid_int = int(sid)
            if sid_int not in by_station:
                continue
            sub = dfm[dfm[STATION_COL_RAW] == sid].copy()
            by_station[sid_int].append(sub)

    out: Dict[int, pd.DataFrame] = {}
    for sid, parts in by_station.items():
        if not parts:
            out[sid] = pd.DataFrame(columns=[TIME_COL, PWR_COL, WSP_COL, WDIR_COL, NAC_COL])
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df.drop_duplicates(subset=[TIME_COL]).sort_values(TIME_COL)
        out[sid] = df[[TIME_COL, PWR_COL, WSP_COL, WDIR_COL, NAC_COL]].copy()

    return out


# =========================
# Main
# =========================
def main():
    meta_path = os.path.join(RAW_DIR, META_FILE)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    mdf = pd.read_csv(meta_path)
    mdf.columns = [_strip_col(c) for c in mdf.columns]
    for col in ("Turbine Name", "Station ID", "Latitude", "Longitude"):
        if col not in mdf.columns:
            raise KeyError(f"[metadata] missing column '{col}'. got={list(mdf.columns)[:30]} ...")

    turbine_names = sorted(mdf["Turbine Name"].astype(str).str.strip().unique().tolist(), key=turbine_sort_key)
    N = len(turbine_names)
    print(f"[dataset3][onlySCADA] turbines from metadata: N={N}, first={turbine_names[:5]} ... last={turbine_names[-3:]}")

    A, latlon, xy_norm, station_ids = build_adj_and_xy_from_metadata(meta_path, turbine_names)
    station_keep = set(int(x) for x in station_ids.tolist())

    # ---- load monthly SCADA ----
    scada_by_station = load_scada_all_years(RAW_DIR, station_keep=station_keep)

    # station_id -> turbine_name mapping (aligned with metadata order)
    station_to_tname = {int(sid): tname for tname, sid in zip(turbine_names, station_ids)}

    turb_raw: Dict[str, pd.DataFrame] = {}
    for sid in station_ids.tolist():
        sid_int = int(sid)
        tname = station_to_tname[sid_int]

        df = scada_by_station.get(sid_int, None)
        if df is None or len(df) == 0:
            raise RuntimeError(f"No SCADA rows for turbine {tname} (StationId={sid_int}). Check your monthly files.")
        df = df.copy()
        df[TIME_COL] = _to_datetime_mixed(df[TIME_COL])
        df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)

        # keep only target years
        df = df[df[TIME_COL].dt.year.isin(ALL_YEARS)].copy()
        turb_raw[tname] = df

    # ---- unified time axis ----
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
        raise RuntimeError("Assume X_STEP_HOURS=3, Y_STEP_HOURS=1 (ratio=3).")

    # split indices (same rule)
    x_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    x_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    y_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_y0, Y_STEP_HOURS)
    y_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_y0, Y_STEP_HOURS)

    # ---- build Y(1h) and X(3h) ----
    P1_list, M1_list = [], []
    X_list, Xv_list = [], []

    for tname in turbine_names:
        df = turb_raw[tname].copy()
        dft = df.set_index(TIME_COL).sort_index()

        # ===== 1h: exact hour observation =====
        sub_hour = dft[[PWR_COL, WSP_COL, WDIR_COL, NAC_COL]].copy()
        sub_hour = sub_hour[(sub_hour.index.minute == 0) & (sub_hour.index.second == 0)]
        sub_hour = sub_hour.groupby(level=0).mean(numeric_only=True)
        sub_hour = sub_hour.reindex(full_time_y)

        P1_raw = sub_hour[PWR_COL].to_numpy(dtype=np.float32)
        W1_raw = sub_hour[WSP_COL].to_numpy(dtype=np.float32)
        Dir1_raw = sub_hour[WDIR_COL].to_numpy(dtype=np.float32)
        Nac1_raw = sub_hour[NAC_COL].to_numpy(dtype=np.float32)

        # dataset3: no LOST -> mask only depends on finite power
        mask_y = np.isfinite(P1_raw).astype(np.float32)

        P1 = np.nan_to_num(P1_raw, nan=0.0).astype(np.float32)
        P1[mask_y < 0.5] = 0.0

        P1_list.append(P1)
        M1_list.append(mask_y)

        # ===== 3h X: take values on full_time_x =====
        P3_raw   = pd.Series(P1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        W3_raw   = pd.Series(W1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Dir3_raw = pd.Series(Dir1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Nac3_raw = pd.Series(Nac1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)

        # valid masks: power validity only depends on finite (no lost filter)
        p3_valid  = np.isfinite(P3_raw).astype(np.float32)
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

        # angle invalid -> set to 0 to avoid fake signal
        dir_sin[dir_valid < 0.5] = 0.0
        dir_cos[dir_valid < 0.5] = 0.0
        nac_sin[nac_valid < 0.5] = 0.0
        nac_cos[nac_valid < 0.5] = 0.0

        # dP: only when consecutive P3 are valid
        dP = np.diff(P3, axis=0, prepend=P3[[0]])
        prev = np.roll(p3_valid, 1); prev[0] = 0.0
        dp_valid = ((p3_valid > 0.5) & (prev > 0.5)).astype(np.float32)
        dP[dp_valid < 0.5] = 0.0

        # P3 invalid -> 0
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

    # stack arrays
    X_raw   = np.stack(X_list, axis=1).astype(np.float32)    # (T3,N,7)
    X_valid = np.stack(Xv_list, axis=1).astype(np.float32)   # (T3,N,7)
    Y_raw   = np.stack(P1_list, axis=1).astype(np.float32)   # (T1,N)
    mask_y  = np.stack(M1_list, axis=1).astype(np.float32)   # (T1,N)

    T3, N2, F = X_raw.shape
    T1 = Y_raw.shape[0]
    assert N2 == N

    # ---- standardization (train only, valid only) ----
    x_train_end = max(0, min(int(x_2019), T3))
    y_train_end = max(0, min(int(y_2019), T1))

    x_mu = np.zeros((F,), dtype=np.float32)
    x_sd = np.ones((F,), dtype=np.float32)

    Xtr = X_raw[:x_train_end]
    Vtr = (X_valid[:x_train_end] > 0.5)
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
    Mtr = (mask_y[:y_train_end] > 0.5)
    if np.any(Mtr):
        y_mu = float(Ytr[Mtr].mean())
        y_sd = float(Ytr[Mtr].std()) + 1e-6
    else:
        y_mu, y_sd = 0.0, 1.0

    Xn = (X_raw - x_mu) / x_sd
    Yn = (Y_raw - y_mu) / y_sd

    Xn[X_valid < 0.5] = 0.0
    Yn[mask_y < 0.5] = 0.0

    # ---- save ----
    np.save(f"{OUT_DIR}/X.npy", Xn.astype(np.float32))
    np.save(f"{OUT_DIR}/X_valid.npy", X_valid.astype(np.float32))
    np.save(f"{OUT_DIR}/Y.npy", Yn.astype(np.float32))
    np.save(f"{OUT_DIR}/mask.npy", mask_y.astype(np.float32))
    np.save(f"{OUT_DIR}/adj.npy", A.astype(np.float32))

    feature_names = ["P3","dP3","W3","dir_sin3","dir_cos3","nac_sin3","nac_cos3"]

    meta = {
        "dataset": "dataset3_hill_of_towie_onlyscada",
        "turbine_ids": turbine_names,          # list[str]
        "station_ids": station_ids.tolist(),   # list[int] aligned with turbine_ids
        "turbine_latlon": latlon.tolist(),     # (N,2)
        "turbine_xy": xy_norm.tolist(),        # (N,2) (optional but useful)
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

        # dataset3 has no lost
        "lost_col": None,
        "lost_rule": None,

        "x_start": str(t_x0),
        "x_end": str(t_x1),
        "y_start": str(t_y0),
        "y_end": str(t_y1),

        "splits": {
            "train_x": [0, int(x_2019)],
            "val_x":   [int(x_2019), int(x_2020)],
            "test_x":  [int(x_2020), int(T3)],
            "train_y": [0, int(y_2019)],
            "val_y":   [int(y_2019), int(y_2020)],
            "test_y":  [int(y_2020), int(T1)],
        },

        "scada_cols": {
            "time": TIME_COL_RAW,
            "station": STATION_COL_RAW,
            "power": RAW_PWR_COL,
            "wind_speed": RAW_WSP_COL,
            "wind_direction_substitute": RAW_WDIR_SUB,
            "nacelle_position": RAW_NAC_COL,
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
