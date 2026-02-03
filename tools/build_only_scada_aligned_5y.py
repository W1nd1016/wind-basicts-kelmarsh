# tools/build_only_scada_5y_match_scada_cerra_rules.py
import os, re, json, glob
import numpy as np
import pandas as pd

RAW_DIR = "raw_scada"
OUT_DIR = "data/wind_onlyscada_no_5b"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_COL = "# Date and time"
PWR_COL  = "Power (kW)"
WSP_COL  = "Wind speed (m/s)"
WDIR_COL = "Wind direction (°)"
NAC_COL  = "Nacelle position (°)"
LOST_COL = "Lost Production to Downtime and Curtailment Total (kWh)"

# ===== 多时间尺度设置 =====
X_STEP_HOURS = 3   # 输入特征：3小时一步（整点观测：从1h序列在0/3/6/...取值）
Y_STEP_HOURS = 1   # 标签：1小时一步（整点观测：取hh:00:00）

H = 6              # 未来6小时（每小时一个点）
L = 9              # 3h一步覆盖24h -> (L-1)*3=24h

# ===== 年份划分（最终要用这套）=====
TRAIN_YEARS = {2016, 2017, 2018}
VAL_YEARS   = {2019}
TEST_YEARS  = {2020}
ALL_YEARS   = TRAIN_YEARS | VAL_YEARS | TEST_YEARS

def turbine_id_from_filename(fname: str):
    # Turbine_Data_Kelmarsh_3_2016-01-03_-_2017-01-01_230.csv
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
    """
    你的规则：
      NaN -> exclude
      0   -> include
      >0  -> exclude
    """
    finite = np.isfinite(arr)
    ok = finite & (arr == 0)
    return ok.astype(np.float32)

def idx_at(ts: pd.Timestamp, base: pd.Timestamp, step_h: int) -> int:
    return int((ts - base) / pd.Timedelta(hours=step_h))

def main():
    # -------- 收集文件：按turbine分组，拼接多年的scada --------
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
        raise RuntimeError(f"期望6个风机，但解析到 {len(turbine_ids)} 个: {turbine_ids}")

    turb_raw = {}
    for tid in turbine_ids:
        dfs = []
        for f in sorted(by_tid[tid]):
            df = read_one_scada_file(f)
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=[TIME_COL]).sort_values(TIME_COL)

        # 只保留 2016-2020
        df_all = df_all[df_all[TIME_COL].dt.year.isin(ALL_YEARS)].copy()
        turb_raw[tid] = df_all

    # -------- 统一时间轴：Y(1h) / X(3h) --------
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

    # 年份边界（用于 meta splits & 标准化范围）
    x_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    x_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    y_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_y0, Y_STEP_HOURS)
    y_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_y0, Y_STEP_HOURS)

    # -------- 构造 Y(1h) 与 X(3h)：整点观测（当前时刻）--------
    P1_list, M1_list = [], []
    X_list, Xv_list = [], []

    for tid in turbine_ids:
        df = turb_raw[tid].copy()
        dft = df.set_index(TIME_COL).sort_index()

        # ====== 1h：整点观测（hh:00:00） ======
        # 功率/风速/风向/机舱角：只取整点观测；同一时刻若重复则取mean
        sub_hour = dft[[PWR_COL, WSP_COL, WDIR_COL, NAC_COL]].copy()
        sub_hour = sub_hour[(sub_hour.index.minute == 0) & (sub_hour.index.second == 0)]
        sub_hour = sub_hour.groupby(level=0).mean(numeric_only=True)
        sub_hour = sub_hour.reindex(full_time_y)  # 对齐到1h网格（缺失 -> NaN）

        # Lost：仍用每小时累计（更稳；用于mask）
        d1_lost = dft[[LOST_COL]].resample("1h").sum(min_count=1).reindex(full_time_y)

        P1_raw = sub_hour[PWR_COL].to_numpy(dtype=np.float32)
        lost1  = d1_lost[LOST_COL].to_numpy(dtype=np.float32)

        ok_lost1  = is_ok_lost(lost1) > 0.5
        p1_finite = np.isfinite(P1_raw)
        mask_y = (p1_finite & ok_lost1).astype(np.float32)

        P1 = np.nan_to_num(P1_raw, nan=0.0).astype(np.float32)
        P1[mask_y < 0.5] = 0.0

        P1_list.append(P1)
        M1_list.append(mask_y)

        # ====== 3h：输入X（整点观测：从1h序列在0/3/6/...取值）=====
        W1_raw   = sub_hour[WSP_COL].to_numpy(dtype=np.float32)
        Dir1_raw = sub_hour[WDIR_COL].to_numpy(dtype=np.float32)
        Nac1_raw = sub_hour[NAC_COL].to_numpy(dtype=np.float32)

        # 直接在 full_time_x 取值（当前时刻观测）
        P3_raw   = pd.Series(P1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        W3_raw   = pd.Series(W1_raw,   index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Dir3_raw = pd.Series(Dir1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Nac3_raw = pd.Series(Nac1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)

        # 3h 的 lost：用过去3小时 rolling sum（严格不向未来看）
        lost1_s = pd.Series(lost1, index=full_time_y)
        finite = np.isfinite(lost1_s.to_numpy(dtype=np.float64)).astype(np.float64)
        s0 = pd.Series(np.nan_to_num(lost1_s.to_numpy(dtype=np.float64), nan=0.0), index=full_time_y)

        sum3 = s0.rolling(window=3, min_periods=3).sum()
        cnt3 = pd.Series(finite, index=full_time_y).rolling(window=3, min_periods=3).sum()

        lost3_s = sum3.where(cnt3 == 3.0, np.nan)  # 窗口内有NaN -> 置NaN
        lost3 = lost3_s.reindex(full_time_x).to_numpy(dtype=np.float32)

        ok_lost3 = is_ok_lost(lost3) > 0.5

        # P3/dP3 受 Lost 过滤；其他特征只看是否finite（与scada+cerra版一致）
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

        # 角度缺失时 sin/cos 置0（避免 NaN -> 0deg -> cos=1 的假信号）
        dir_sin[dir_valid < 0.5] = 0.0
        dir_cos[dir_valid < 0.5] = 0.0
        nac_sin[nac_valid < 0.5] = 0.0
        nac_cos[nac_valid < 0.5] = 0.0

        # dP：只在连续两帧 P3 都有效（含未限电）才计算，否则置0
        dP = np.diff(P3, axis=0, prepend=P3[[0]])
        prev = np.roll(p3_valid, 1); prev[0] = 0.0
        dp_valid = ((p3_valid > 0.5) & (prev > 0.5)).astype(np.float32)
        dP[dp_valid < 0.5] = 0.0

        # P3 无效也置0
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

    # (T3,N,F) / (T1,N)
    X_raw   = np.stack(X_list, axis=1).astype(np.float32)   # (T3,N,7)
    X_valid = np.stack(Xv_list, axis=1).astype(np.float32)
    Y_raw   = np.stack(P1_list, axis=1).astype(np.float32)  # (T1,N)
    mask_y  = np.stack(M1_list, axis=1).astype(np.float32)

    T3, N, F = X_raw.shape
    T1 = Y_raw.shape[0]

    # -------- 标准化：只用训练年 & 只统计 valid 点 --------
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
            x_mu[f] = float(vals.mean())
            x_sd[f] = float(vals.std()) + 1e-6
        else:
            x_mu[f] = 0.0
            x_sd[f] = 1.0

    # Y 标准化：只用训练年且mask=1
    Ytr = Y_raw[:y_train_end]
    Mtr = mask_y[:y_train_end] > 0.5
    if np.any(Mtr):
        y_mu = float(Ytr[Mtr].mean())
        y_sd = float(Ytr[Mtr].std()) + 1e-6
    else:
        y_mu, y_sd = 0.0, 1.0

    Xn = (X_raw - x_mu) / x_sd
    Yn = (Y_raw - y_mu) / y_sd

    # invalid 强制置0
    Xn[X_valid < 0.5] = 0.0
    Yn[mask_y < 0.5] = 0.0

    # -------- 邻接矩阵 --------
    static_path = os.path.join(RAW_DIR, "Kelmarsh_WT_static.csv")
    if os.path.exists(static_path):
        A = build_adj_from_static(static_path, turbine_ids)
    else:
        A = np.ones((N, N), dtype=np.float32) / N

    # 保存
    np.save(f"{OUT_DIR}/X.npy", Xn.astype(np.float32))              # (T3,N,7)
    np.save(f"{OUT_DIR}/X_valid.npy", X_valid.astype(np.float32))  # (T3,N,7)
    np.save(f"{OUT_DIR}/Y.npy", Yn.astype(np.float32))             # (T1,N)
    np.save(f"{OUT_DIR}/mask.npy", mask_y.astype(np.float32))      # (T1,N)
    np.save(f"{OUT_DIR}/adj.npy", A.astype(np.float32))

    feature_names = ["P3","dP3","W3","dir_sin3","dir_cos3","nac_sin3","nac_cos3"]

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
