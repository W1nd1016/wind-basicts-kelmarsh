# tools/build_scada_cerra_v1_s2_FnpOnlyAnalysis_dataset3.py
import os, re, json, glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# Config
# =========================
RAW_DIR   = "raw_scada_dataset3"
CERRA_DIR = "raw_new_cerra1hour_dataset3"
OUT_DIR   = "data/wind_scada_cerra_v1_s2_FnpOnlyAnalysis_dataset3"
os.makedirs(OUT_DIR, exist_ok=True)

SCADA_TURBINE_PATTERN = "tblSCTurbine_*.csv"
SCADA_GRID_PATTERN    = "tblSCTurGrid_*.csv"
META_FILE = "Hill_of_Towie_turbine_metadata.csv"

# Canonical internal names (keep dataset1 style)
TIME_COL = "TimeStamp"
STATION_COL = "StationId"

PWR_COL  = "Power (kW)"
WSP_COL  = "Wind speed (m/s)"
WDIR_COL = "Wind direction (°)"
NAC_COL  = "Nacelle position (°)"

# Time settings (same as dataset1/paper)
X_STEP_HOURS = 3
Y_STEP_HOURS = 1
H = 6
L = 9

TRAIN_YEARS = {2016, 2017, 2018}
VAL_YEARS   = {2019}
TEST_YEARS  = {2020}
ALL_YEARS   = TRAIN_YEARS | VAL_YEARS | TEST_YEARS

# CERRA
FORECAST_LEADS = [1, 2, 3, 4, 5, 6]
K_NEIGHBORS = 16
CERRA_VAR_ORDER = ["speed", "direction", "u", "v"]


# =========================
# Small utils
# =========================
def _strip_col(c: str) -> str:
    return str(c).strip().lstrip("\ufeff")


def idx_at(ts: pd.Timestamp, base: pd.Timestamp, step_h: int) -> int:
    return int((ts - base) / pd.Timedelta(hours=step_h))


def turbine_sort_key(name: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", str(name))
    if m:
        return (int(m.group(1)), str(name))
    return (10**9, str(name))


def cerra_name_candidates(tname: str) -> List[str]:
    """
    Try both T01 and T1 styles.
    """
    tname = str(tname)
    m = re.match(r"^T(\d+)$", tname)
    if not m:
        return [tname]
    k = m.group(1)
    k2 = str(int(k))
    if k2 == k:
        return [tname]
    return [tname, f"T{k2}"]


def _read_header_cols(path: str) -> List[str]:
    df0 = pd.read_csv(path, nrows=0)
    return [_strip_col(c) for c in df0.columns]


def _read_csv_by_needed(path: str, needed_stripped: List[str]) -> pd.DataFrame:
    need_set = set(needed_stripped)

    def usecols(c):
        return _strip_col(c) in need_set

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.columns = [_strip_col(c) for c in df.columns]
    miss = need_set - set(df.columns)
    if miss:
        raise KeyError(f"{os.path.basename(path)} missing columns after read: {sorted(miss)}")
    return df

def _to_datetime_mixed(s: pd.Series) -> pd.Series:
    """
    Robust datetime parser for mixed formats like:
      '2017-07-01 00:10:00' and '2017-07-01'
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    # 关键：cache=False 防止 pandas 锁死格式导致报错
    try:
        # pandas>=2.0 支持 format="mixed"
        return pd.to_datetime(s, errors="coerce", format="mixed", cache=False)
    except TypeError:
        # older pandas fallback
        return pd.to_datetime(s, errors="coerce", cache=False)


def _pick_by_regex(cols: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rgx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rgx.search(c):
                return c
    return None


def _list_matches(cols: List[str], pat: str) -> List[str]:
    rgx = re.compile(pat, flags=re.IGNORECASE)
    return [c for c in cols if rgx.search(c)]


# =========================
# Detect SCADA columns (dataset3)
# =========================
def detect_turbine_cols(cols: List[str]) -> Tuple[str, str, Optional[str]]:
    """
    返回 (wsp_col, yawpos_col, nacelpos_col)

    规则：
      - 风速：优先 wtc_AcWindSp_mean / *WindSp*_mean / wtc_SecAnemo_mean
      - 风向（替代）：强制用 wtc_YawPos_mean
      - Nacelle position：优先 wtc_NacelPos_mean；若找不到则返回 None（后续用 yawpos 兜底）
    """
    # wind speed
    wsp = _pick_by_regex(cols, [
        r"^wtc_.*acwindsp.*_mean$",
        r"^wtc_.*windsp.*_mean$",
        r"^wtc_secAnemo_mean$",     
        r"^wtc_.*anemo.*_mean$",
    ])
    if wsp is None:
        hints = _list_matches(cols, r"anemo|windsp")
        raise KeyError(
            "Cannot find wind speed mean column in tblSCTurbine. "
            f"Examples of possible matches: {hints[:30]}"
        )

    # wind direction substitute: yaw position mean (required)
    yaw = _pick_by_regex(cols, [
        r"^wtc_yawpos_mean$",
        r"^wtc_.*yawpos.*_mean$",
    ])
    if yaw is None:
        hints = _list_matches(cols, r"yawpos|yaw")
        raise KeyError(
            "Cannot find wtc_YawPos_mean (or yawpos mean) in tblSCTurbine. "
            f"Possible matches: {hints[:30]}"
        )

    # nacelle position: prefer wtc_NacelPos_mean, allow fallback to yaw later
    nac = _pick_by_regex(cols, [
        r"^wtc_nacelpos_mean$",
        r"^wtc_.*nacelpos.*_mean$",
        r"^wtc_.*nacel.*pos.*_mean$",
        r"^wtc_.*nac.*pos.*_mean$",
    ])
    # nac can be None (we will fallback in read_one_month_scada)

    return wsp, yaw, nac



def detect_grid_power_col(cols: List[str]) -> str:
    """
    Returns power column from tblSCTurGrid file columns.
    Prefer ActPower_mean.
    """
    pwr = _pick_by_regex(cols, [
        r"^wtc_.*actpower.*_mean$",
        r"^wtc_.*actualpower.*_mean$",
        r"^wtc_.*act.*power.*_mean$",
        r"^wtc_.*power.*_mean$",
    ])
    if pwr is None:
        hints = _list_matches(cols, r"power")
        raise KeyError(
            "Cannot find power mean column in tblSCTurGrid. "
            f"Columns containing 'power': {hints[:30]}"
        )
    return pwr


# =========================
# SCADA load (monthly files)
# =========================
def parse_year_month_from_fname(fname: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"_(\d{4})_(\d{2})\.csv$", fname)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def read_one_month_scada(turbine_csv: Optional[str], grid_csv: Optional[str], station_keep: set,
                         verbose: bool = False) -> pd.DataFrame:
    """
    Return merged df with canonical columns:
      TimeStamp, StationId, Power (kW), Wind speed (m/s), Wind direction (°), Nacelle position (°)
    """
        # ---- turbine side (wsp + yawpos + nacelpos) ----
    if turbine_csv is not None:
        cols = _read_header_cols(turbine_csv)
        wsp_c, yaw_c, nac_c = detect_turbine_cols(cols)

        needed = [TIME_COL, STATION_COL, wsp_c, yaw_c]
        if nac_c is not None:
            needed.append(nac_c)

        dt = _read_csv_by_needed(turbine_csv, needed)

        # rename to canonical
        dt = dt.rename(columns={wsp_c: WSP_COL})

        # 1) Wind direction (°) 强制用 yawpos
        dt = dt.rename(columns={yaw_c: WDIR_COL})

        # 2) Nacelle position (°) 用 nacelpos；若不存在则用 yawpos 兜底
        if nac_c is not None:
            dt = dt.rename(columns={nac_c: NAC_COL})
        else:
            dt[NAC_COL] = dt[WDIR_COL]
            if verbose:
                print(f"[WARN] {os.path.basename(turbine_csv)}: missing nacelle-pos col (wtc_NacelPos_mean). Fallback NAC=YawPos.")

        # numeric coercion
        for c in [WSP_COL, WDIR_COL, NAC_COL]:
            dt[c] = pd.to_numeric(dt[c], errors="coerce")

        dt[TIME_COL] = _to_datetime_mixed(dt[TIME_COL])
        dt = dt.dropna(subset=[TIME_COL])

        dt[STATION_COL] = pd.to_numeric(dt[STATION_COL], errors="coerce").astype("Int64")
        dt = dt.dropna(subset=[TIME_COL, STATION_COL])
        dt = dt[dt[STATION_COL].isin(list(station_keep))]
        dt = dt.groupby([TIME_COL, STATION_COL], as_index=False).mean(numeric_only=True)

        if verbose:
            print(
                f"[SCADA turb] {os.path.basename(turbine_csv)} -> "
                f"WSP from {wsp_c}, WDIR(from yaw)={yaw_c}, NAC={(nac_c if nac_c is not None else 'fallback=yaw')}"
            )
    else:
        dt = pd.DataFrame(columns=[TIME_COL, STATION_COL, WSP_COL, WDIR_COL, NAC_COL])


    # ---- grid side (power) ----
    if grid_csv is not None:
        cols = _read_header_cols(grid_csv)
        pwr_c = detect_grid_power_col(cols)

        dg = _read_csv_by_needed(grid_csv, [TIME_COL, STATION_COL, pwr_c])
        dg = dg.rename(columns={pwr_c: PWR_COL})
        dg[PWR_COL] = pd.to_numeric(dg[PWR_COL], errors="coerce")

        dg[TIME_COL] = _to_datetime_mixed(dg[TIME_COL])
        dg = dg.dropna(subset=[TIME_COL])

        dg[STATION_COL] = pd.to_numeric(dg[STATION_COL], errors="coerce").astype("Int64")
        dg = dg.dropna(subset=[TIME_COL, STATION_COL])
        dg = dg[dg[STATION_COL].isin(list(station_keep))]
        dg = dg.groupby([TIME_COL, STATION_COL], as_index=False).mean(numeric_only=True)

        if verbose:
            print(f"[SCADA grid] {os.path.basename(grid_csv)} -> power={PWR_COL} from {pwr_c}")

    else:
        dg = pd.DataFrame(columns=[TIME_COL, STATION_COL, PWR_COL])

    merged = pd.merge(dt, dg, on=[TIME_COL, STATION_COL], how="outer")
    merged = merged.sort_values([TIME_COL, STATION_COL])
    return merged


def load_scada_all_years(raw_dir: str, station_keep: set) -> Dict[int, pd.DataFrame]:
    turb_files = sorted(glob.glob(os.path.join(raw_dir, SCADA_TURBINE_PATTERN)))
    grid_files = sorted(glob.glob(os.path.join(raw_dir, SCADA_GRID_PATTERN)))

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
        raise FileNotFoundError(f"No SCADA monthly files found in {raw_dir} for years={sorted(ALL_YEARS)}")

    by_station: Dict[int, List[pd.DataFrame]] = {int(s): [] for s in station_keep}

    for i, (y, m) in enumerate(keys):
        tpath = turb_map.get((y, m), None)
        gpath = grid_map.get((y, m), None)
        if tpath is None and gpath is None:
            continue

        # verbose for first few months to show detected columns
        verbose = (i < 3)

        dfm = read_one_month_scada(tpath, gpath, station_keep=station_keep, verbose=verbose)
        if len(dfm) == 0:
            continue

        for sid in dfm[STATION_COL].dropna().unique():
            sid_int = int(sid)
            if sid_int not in by_station:
                continue
            sub = dfm[dfm[STATION_COL] == sid].copy()
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
# Metadata -> adj/xy
# =========================
def build_adj_and_xy_from_metadata(metadata_path: str, turbine_names: List[str]):
    df = pd.read_csv(metadata_path)
    df.columns = [_strip_col(c) for c in df.columns]

    need = {"Turbine Name", "Station ID", "Latitude", "Longitude"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"[metadata] Missing columns: {sorted(miss)}")

    df["Turbine Name"] = df["Turbine Name"].astype(str).str.strip()
    df["Station ID"] = df["Station ID"].astype(int)

    df = df[df["Turbine Name"].isin(turbine_names)].copy()
    df["__ord"] = df["Turbine Name"].apply(lambda x: turbine_names.index(x))
    df = df.sort_values("__ord")

    coords = df[["Latitude", "Longitude"]].values.astype(np.float32)  # lat, lon
    station_ids = df["Station ID"].to_numpy(dtype=np.int64)

    lat0 = float(coords[:, 0].mean())
    scale_lat = 111000.0
    scale_lon = 111000.0 * np.cos(np.deg2rad(lat0))

    xy = np.stack([
        (coords[:, 1] - coords[:, 1].mean()) * scale_lon,
        (coords[:, 0] - coords[:, 0].mean()) * scale_lat
    ], axis=1).astype(np.float32)

    N = xy.shape[0]
    d = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            d[i, j] = np.linalg.norm(xy[i] - xy[j])

    sigma = np.median(d[d > 0]) if np.any(d[d > 0]) else 1.0
    A = np.exp(-(d ** 2) / (sigma ** 2)).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-6)

    xy_norm = xy.copy()
    s = np.std(xy_norm, axis=0, keepdims=True) + 1e-6
    xy_norm = (xy_norm / s).astype(np.float32)

    return A.astype(np.float32), coords.astype(np.float32), xy_norm.astype(np.float32), station_ids


# =========================
# CERRA load (same as dataset1 logic, only name changes)
# =========================
def parse_year_from_cerra_fname(fname: str) -> Optional[int]:
    m = re.search(r"_(\d{4})\.csv$", fname)
    return int(m.group(1)) if m else None


def _cerra_year_paths(tname: str) -> List[str]:
    paths = []
    for cand in cerra_name_candidates(tname):
        paths.extend(glob.glob(os.path.join(CERRA_DIR, f"CERRA_75m_{cand}_*.csv")))
    return sorted(list(set(paths)))


def load_neighbors_pos(tname: str) -> np.ndarray:
    npz_path = None
    for cand in cerra_name_candidates(tname):
        p = os.path.join(CERRA_DIR, f"CERRA_75m_{cand}_neighbors.npz")
        if os.path.exists(p):
            npz_path = p
            break
    if npz_path is None:
        raise FileNotFoundError(f"Missing neighbors npz for {tname}")

    z = np.load(npz_path, allow_pickle=True)
    dx_m = z["dx_m"].astype(np.float32)
    dy_m = z["dy_m"].astype(np.float32)
    dist_m = z["dist_m"].astype(np.float32)

    dx_scale = float(np.max(np.abs(dx_m)) + 1e-6)
    dy_scale = float(np.max(np.abs(dy_m)) + 1e-6)
    dist_scale = float(np.max(dist_m) + 1e-6)

    dx = (dx_m / dx_scale).astype(np.float32)
    dy = (dy_m / dy_scale).astype(np.float32)
    ds = (dist_m / dist_scale).astype(np.float32)

    pos = np.stack([dx, dy, ds], axis=-1).astype(np.float32)
    if pos.shape[0] != K_NEIGHBORS:
        raise RuntimeError(f"[{tname}] K mismatch in neighbors.npz: expected {K_NEIGHBORS}, got {pos.shape[0]}")
    return pos


def read_cerra_analysis_all_years(tname: str, full_time_x: pd.DatetimeIndex):
    paths = _cerra_year_paths(tname)
    if not paths:
        raise FileNotFoundError(f"[{tname}] No CERRA year files found in {CERRA_DIR}")

    dfs = []
    for p in paths:
        y = parse_year_from_cerra_fname(os.path.basename(p))
        if y is None or y not in ALL_YEARS:
            continue
        df = pd.read_csv(p)
        if "time" not in df.columns:
            raise KeyError(f"{p} missing 'time'")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"[{tname}] No CERRA rows for years={sorted(ALL_YEARS)}")

    df_all = pd.concat(dfs, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
    df_all = df_all.reindex(full_time_x)

    cols = []
    for k in range(1, K_NEIGHBORS + 1):
        for v in CERRA_VAR_ORDER:
            cols.append(f"{v}_k{k}")

    miss = [c for c in cols if c not in df_all.columns]
    if miss:
        raise KeyError(f"[{tname}] missing analysis cols: {miss[:8]} ... total_missing={len(miss)}")

    sub = df_all[cols].copy()
    valid = np.isfinite(sub.values).astype(np.float32)
    sub = sub.fillna(0.0)
    names = [f"{c}_75" for c in cols]
    return sub.values.astype(np.float32), valid.astype(np.float32), names


def read_cerra_forecast_all_years(tname: str, full_time_x: pd.DatetimeIndex):
    paths = _cerra_year_paths(tname)
    if not paths:
        raise FileNotFoundError(f"[{tname}] No CERRA year files found in {CERRA_DIR}")

    dfs = []
    for p in paths:
        y = parse_year_from_cerra_fname(os.path.basename(p))
        if y is None or y not in ALL_YEARS:
            continue
        df = pd.read_csv(p)
        if "time" not in df.columns:
            raise KeyError(f"{p} missing 'time'")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"[{tname}] No CERRA rows for years={sorted(ALL_YEARS)}")

    df_all = pd.concat(dfs, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
    df_all = df_all.reindex(full_time_x)

    T3 = len(full_time_x)
    fc = np.zeros((T3, H, K_NEIGHBORS, 4), dtype=np.float32)
    fv = np.zeros((T3, H, K_NEIGHBORS, 4), dtype=np.float32)

    for li, lead in enumerate(FORECAST_LEADS[:H]):
        for k in range(1, K_NEIGHBORS + 1):
            for vi, vname in enumerate(CERRA_VAR_ORDER):
                col_main = f"{vname}_k{k}_fc{lead}"
                col_alt  = f"{vname}_k{k}_fc{lead:02d}"
                col = col_main if col_main in df_all.columns else (col_alt if col_alt in df_all.columns else None)
                if col is None:
                    raise KeyError(f"[{tname}] missing forecast col: {col_main} (also tried {col_alt})")
                arr = df_all[col].to_numpy(dtype=np.float32)
                ok = np.isfinite(arr).astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
                fc[:, li, k-1, vi] = arr
                fv[:, li, k-1, vi] = ok

    return fc.astype(np.float32), fv.astype(np.float32)


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

    mdf["Turbine Name"] = mdf["Turbine Name"].astype(str).str.strip()
    turbine_names = sorted(mdf["Turbine Name"].unique().tolist(), key=turbine_sort_key)
    N = len(turbine_names)
    print(f"[dataset3] turbines from metadata: N={N}, first={turbine_names[:5]} ... last={turbine_names[-3:]}")

    A, latlon, xy_norm, station_ids = build_adj_and_xy_from_metadata(meta_path, turbine_names)
    station_keep = set(int(x) for x in station_ids.tolist())

    # ---- SCADA ----
    scada_by_station = load_scada_all_years(RAW_DIR, station_keep=station_keep)
    station_to_tname = {int(sid): tname for tname, sid in zip(turbine_names, station_ids)}

    turb_raw: Dict[str, pd.DataFrame] = {}
    for sid in station_ids.tolist():
        sid_int = int(sid)
        tname = station_to_tname[sid_int]
        df = scada_by_station.get(sid_int, None)
        if df is None or len(df) == 0:
            raise RuntimeError(f"No SCADA data for turbine {tname} (StationId={sid_int}).")
        df = df.copy()
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL)
        df = df[df[TIME_COL].dt.year.isin(ALL_YEARS)].copy()
        turb_raw[tname] = df

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
    if ratio != 3:
        raise RuntimeError("Assume X_STEP_HOURS=3 and Y_STEP_HOURS=1.")

    x_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    x_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    y_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_y0, Y_STEP_HOURS)
    y_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_y0, Y_STEP_HOURS)

    # ---- build arrays ----
    P1_list, M1_list = [], []
    X_list, Xv_list = [], []
    AN_list, ANv_list = [], []
    FC_list, FCv_list = [], []
    pos_list = []
    cerra_an_feature_names = None

    for tname in turbine_names:
        df = turb_raw[tname].copy()
        dft = df.set_index(TIME_COL).sort_index()

        # hourly sample (same rule)
        sub_hour = dft[[PWR_COL, WSP_COL, WDIR_COL, NAC_COL]].copy()
        sub_hour = sub_hour[(sub_hour.index.minute == 0) & (sub_hour.index.second == 0)]
        sub_hour = sub_hour.groupby(level=0).mean(numeric_only=True)
        sub_hour = sub_hour.reindex(full_time_y)

        P1_raw   = sub_hour[PWR_COL].to_numpy(dtype=np.float32)
        W1_raw   = sub_hour[WSP_COL].to_numpy(dtype=np.float32)
        Dir1_raw = sub_hour[WDIR_COL].to_numpy(dtype=np.float32)
        Nac1_raw = sub_hour[NAC_COL].to_numpy(dtype=np.float32)

        mask_y = np.isfinite(P1_raw).astype(np.float32)
        P1 = np.nan_to_num(P1_raw, nan=0.0).astype(np.float32)
        P1[mask_y < 0.5] = 0.0
        P1_list.append(P1)
        M1_list.append(mask_y)

        # 3-hour aligned
        P3_raw   = pd.Series(P1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        W3_raw   = pd.Series(W1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Dir3_raw = pd.Series(Dir1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Nac3_raw = pd.Series(Nac1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)

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

        # CERRA analysis
        AN, ANv, an_names = read_cerra_analysis_all_years(tname, full_time_x)
        if cerra_an_feature_names is None:
            cerra_an_feature_names = an_names
        AN_list.append(AN)
        ANv_list.append(ANv)

        # CERRA forecast
        FC, FCv = read_cerra_forecast_all_years(tname, full_time_x)
        FC_list.append(FC)
        FCv_list.append(FCv)

        # neighbors pos
        pos_list.append(load_neighbors_pos(tname))

    # stack
    X_scada = np.stack(X_list, axis=1).astype(np.float32)      # (T3,N,7)
    Xv_scada = np.stack(Xv_list, axis=1).astype(np.float32)    # (T3,N,7)
    AN_feat = np.stack(AN_list, axis=1).astype(np.float32)     # (T3,N,K*4)
    AN_valid = np.stack(ANv_list, axis=1).astype(np.float32)   # (T3,N,K*4)
    FC_feat = np.stack(FC_list, axis=2).astype(np.float32)     # (T3,H,N,K,4)
    FC_valid = np.stack(FCv_list, axis=2).astype(np.float32)   # (T3,H,N,K,4)
    pos_all = np.stack(pos_list, axis=0).astype(np.float32)    # (N,K,3)

    X_raw = np.concatenate([X_scada, AN_feat], axis=-1).astype(np.float32)
    X_valid = np.concatenate([Xv_scada, AN_valid], axis=-1).astype(np.float32)

    Y_raw = np.stack(P1_list, axis=1).astype(np.float32)       # (T1,N)
    mask_y = np.stack(M1_list, axis=1).astype(np.float32)      # (T1,N)

    T3, N2, F = X_raw.shape
    T1 = Y_raw.shape[0]
    assert N2 == N

    feature_names = ["P3","dP3","W3","dir_sin3","dir_cos3","nac_sin3","nac_cos3"] + (cerra_an_feature_names or [])

    # ---- normalize on train ----
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

    # ---- normalize FC separately ----
    FCtr = FC_feat[:x_train_end]
    FVtr = (FC_valid[:x_train_end] > 0.5).astype(np.float32)

    eps = 1e-6
    cnt = FVtr.sum(axis=(0, 2)) + eps
    s1 = (FCtr * FVtr).sum(axis=(0, 2))
    s2 = ((FCtr * FVtr) ** 2).sum(axis=(0, 2))

    fc_mu = (s1 / cnt).astype(np.float32)
    var = (s2 / cnt) - (fc_mu ** 2)
    var = np.maximum(var, 0.0).astype(np.float32)
    fc_sd = (np.sqrt(var) + 1e-6).astype(np.float32)

    bad = (cnt <= 1.0) | (fc_sd < 1e-5)
    fc_mu[bad] = 0.0
    fc_sd[bad] = 1.0

    FCn = (FC_feat - fc_mu[None, :, None, :, :]) / fc_sd[None, :, None, :, :]
    FCn[FC_valid < 0.5] = 0.0

    # ---- save ----
    np.save(f"{OUT_DIR}/X.npy", Xn.astype(np.float32))
    np.save(f"{OUT_DIR}/X_valid.npy", X_valid.astype(np.float32))
    np.save(f"{OUT_DIR}/Y.npy", Yn.astype(np.float32))
    np.save(f"{OUT_DIR}/mask.npy", mask_y.astype(np.float32))
    np.save(f"{OUT_DIR}/adj.npy", A.astype(np.float32))

    np.save(f"{OUT_DIR}/FC.npy", FCn.astype(np.float32))
    np.save(f"{OUT_DIR}/FC_valid.npy", FC_valid.astype(np.float32))
    np.save(f"{OUT_DIR}/pos.npy", pos_all.astype(np.float32))

    meta = {
        "dataset": "dataset3_hill_of_towie",
        "turbine_ids": turbine_names,
        "station_ids": station_ids.tolist(),
        "turbine_latlon": latlon.tolist(),
        "turbine_xy": xy_norm.tolist(),
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
        "lost_col": None,
        "lost_rule": None,
        "x_start": str(t_x0),
        "x_end": str(t_x1),
        "y_start": str(t_y0),
        "y_end": str(t_y1),
        "train_years": sorted(list(TRAIN_YEARS)),
        "val_years": sorted(list(VAL_YEARS)),
        "test_years": sorted(list(TEST_YEARS)),
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
            "forecast_shape": [int(H), int(K_NEIGHBORS), 4],
            "pos_shape": [int(K_NEIGHBORS), 3],
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
