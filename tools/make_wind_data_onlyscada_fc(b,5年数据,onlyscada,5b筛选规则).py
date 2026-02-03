# tools/build_only_scada_aligned_5y.py
import os, re, json, glob
import numpy as np
import pandas as pd

RAW_DIR = "raw_scada"
OUT_DIR = "data/wind_onlyscada"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_COL = "# Date and time"
PWR_COL  = "Power (kW)"
WSP_COL  = "Wind speed (m/s)"
WDIR_COL = "Wind direction (°)"
NAC_COL  = "Nacelle position (°)"
LOST_COL = "Lost Production to Downtime and Curtailment Total (kWh)"

# ===== 多时间尺度设置（对齐你的Dataset2风格）=====
X_STEP_HOURS = 3   # 输入特征：3小时一步（直接取00/03/06...这些点）
Y_STEP_HOURS = 1   # 标签：1小时一步（每小时预测）

H = 6              # 未来6小时（每小时一个点）
L = 9              # 3h一步，覆盖24h => (L-1)*3 = 24

# ===== 年份划分 =====
TRAIN_YEARS = {2016, 2017, 2018}
VAL_YEARS   = {2019}
TEST_YEARS  = {2020}
ALL_YEARS   = TRAIN_YEARS | VAL_YEARS | TEST_YEARS

# 每年丢掉1/1-1/2（按你的要求）
DROP_JAN_1_2 = True

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

def mask_drop_jan_1_2(time_index: pd.DatetimeIndex) -> np.ndarray:
    if not DROP_JAN_1_2:
        return np.ones((len(time_index),), dtype=np.float32)
    md = (time_index.month == 1) & (time_index.day <= 2)
    return (~md).astype(np.float32)

def main():
    # -------- 收集文件：按turbine分组，拼接多年的scada --------
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "Turbine_Data_Kelmarsh_*.csv")))
    if not all_files:
        raise FileNotFoundError(f"raw_scada 下找不到 Turbine_Data_Kelmarsh_*.csv")

    by_tid = {}
    for f in all_files:
        tid = turbine_id_from_filename(os.path.basename(f))
        if tid is None:
            continue
        by_tid.setdefault(tid, []).append(f)

    turbine_ids = sorted(by_tid.keys())
    if len(turbine_ids) != 6:
        raise RuntimeError(f"期望6个风机文件组，但解析到 {len(turbine_ids)} 个: {turbine_ids}")

    # 读取并拼接每个turbine的所有年份文件
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

    # -------- 统一时间轴：Y(1h) 从最早到最晚，X(3h) 对齐 --------
    t0 = min(df[TIME_COL].min() for df in turb_raw.values())
    t1 = max(df[TIME_COL].max() for df in turb_raw.values())

    # 为了让 X/Y 映射稳定：对齐到整点
    t_y0 = pd.to_datetime(t0).ceil("1h")
    t_y1 = pd.to_datetime(t1).floor("1h")
    full_time_y = pd.date_range(t_y0, t_y1, freq="1h")

    t_x0 = t_y0.ceil(f"{X_STEP_HOURS}h")
    t_x1 = t_y1.floor(f"{X_STEP_HOURS}h")
    full_time_x = pd.date_range(t_x0, t_x1, freq=f"{X_STEP_HOURS}h")

    ratio = X_STEP_HOURS // Y_STEP_HOURS  # 3
    offset_hours = int((t_x0 - t_y0) / pd.Timedelta(hours=1))
    if ratio != 3:
        raise RuntimeError("当前实现假设 X_STEP_HOURS=3, Y_STEP_HOURS=1")

    # 年份边界（按 index 切分，split 仍然用连续range；跨年H小时的样本会在Dataset里再过滤）
    def idx_at(ts: pd.Timestamp, base: pd.Timestamp, step_h: int):
        return int((ts - base) / pd.Timedelta(hours=step_h))

    x_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    x_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_x0, X_STEP_HOURS)
    y_2019 = idx_at(pd.Timestamp("2019-01-01 00:00:00"), t_y0, Y_STEP_HOURS)
    y_2020 = idx_at(pd.Timestamp("2020-01-01 00:00:00"), t_y0, Y_STEP_HOURS)

    # -------- 构造 Y(1h) 与 X(3h) --------
    P1_list, M1_list = [], []
    X_list, Xv_list = [], []

    # 对“每年丢掉1/1-1/2”
    keep_y = mask_drop_jan_1_2(full_time_y)  # (T1,)
    keep_x = mask_drop_jan_1_2(full_time_x)  # (T3,)

    for tid in turbine_ids:
        df = turb_raw[tid]

        # ====== 1) hourly 点值（按你的要求：直接取整点，不做1h插值）======
        # 先取所有 minute==0 的点作为“hourly观测”
        df_hp = df[df[TIME_COL].dt.minute.eq(0) & df[TIME_COL].dt.second.eq(0)].copy()
        df_hp = df_hp.groupby(TIME_COL).mean(numeric_only=True)  # 去重/聚合
        df_hp = df_hp.reindex(full_time_y)

        P1_raw   = df_hp[PWR_COL].to_numpy(dtype=np.float32)
        W1_raw   = df_hp[WSP_COL].to_numpy(dtype=np.float32)
        Dir1_raw = df_hp[WDIR_COL].to_numpy(dtype=np.float32)
        Nac1_raw = df_hp[NAC_COL].to_numpy(dtype=np.float32)

        # ====== 2) Lost：用“小时累计”最稳妥（即使你文件是10-min，也不会漏判）======
        dfl = df.set_index(TIME_COL)[[LOST_COL]].sort_index()
        lost1 = dfl.resample("1h").sum(min_count=1).reindex(full_time_y)[LOST_COL].to_numpy(dtype=np.float32)

        # ====== 3) 标签mask（1h）=====
        ok_lost1 = is_ok_lost(lost1) > 0.5
        p1_finite = np.isfinite(P1_raw)
        mask_y = (p1_finite & ok_lost1 & (keep_y > 0.5)).astype(np.float32)

        P1 = np.nan_to_num(P1_raw, nan=0.0)
        P1[mask_y < 0.5] = 0.0

        P1_list.append(P1)
        M1_list.append(mask_y)

        # ====== 4) 3h 输入：直接对齐到 full_time_x（从 hourly 点里取 00/03/06...）=====
        # 这里不做3h均值；就是取这些时点的观测
        # 但 Lost(3h) 用3小时累计更合理（只要3小时窗口里发生过限电/停机，就当无效）
        P3_raw   = pd.Series(P1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        W3_raw   = pd.Series(W1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Dir3_raw = pd.Series(Dir1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)
        Nac3_raw = pd.Series(Nac1_raw, index=full_time_y).reindex(full_time_x).to_numpy(dtype=np.float32)

        lost3 = dfl.resample(f"{X_STEP_HOURS}h").sum(min_count=1).reindex(full_time_x)[LOST_COL].to_numpy(dtype=np.float32)

        ok_lost3 = is_ok_lost(lost3) > 0.5

        # ====== 5) X_valid（用于统计均值方差 & 5B anchors）=====
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

        # 缺失角度时，把 sin/cos 也置0，避免“0度 => cos=1”的假信号
        dir_sin[dir_valid < 0.5] = 0.0
        dir_cos[dir_valid < 0.5] = 0.0
        nac_sin[nac_valid < 0.5] = 0.0
        nac_cos[nac_valid < 0.5] = 0.0

        # dP：只在连续两帧 P3 都有效(含未限电)时计算，否则置0
        dP = np.diff(P3, axis=0, prepend=P3[[0]])
        prev = np.roll(p3_valid, 1); prev[0] = 0.0
        dp_valid = ((p3_valid > 0.5) & (prev > 0.5)).astype(np.float32)
        dP[dp_valid < 0.5] = 0.0

        # P3无效也置0
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
    X_raw = np.stack(X_list, axis=1).astype(np.float32)
    X_valid = np.stack(Xv_list, axis=1).astype(np.float32)

    Y_raw = np.stack(P1_list, axis=1).astype(np.float32)
    mask_y = np.stack(M1_list, axis=1).astype(np.float32)

    # -------- 标准化：只用训练年 + 只统计 valid 点（对齐你之前的严谨做法）--------
    # train_x: 2016-01-01..2018-12-31 (在我们的索引上就是 [0, x_2019))
    T3 = X_raw.shape[0]
    T1 = Y_raw.shape[0]
    x_train_end = max(0, min(x_2019, T3))
    y_train_end = max(0, min(y_2019, T1))

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

    # Y标准化：只用训练年且mask=1
    Ytr = Y_raw[:y_train_end]
    Mtr = mask_y[:y_train_end] > 0.5
    if np.any(Mtr):
        y_mu = float(Ytr[Mtr].mean())
        y_sd = float(Ytr[Mtr].std()) + 1e-6
    else:
        y_mu, y_sd = 0.0, 1.0

    Xn = (X_raw - x_mu) / x_sd
    Yn = (Y_raw - y_mu) / y_sd

    # invalid 强制置0（缺失不会污染数值；anchors会按5B过滤）
    Xn[X_valid < 0.5] = 0.0
    Yn[mask_y < 0.5] = 0.0

    # -------- 邻接矩阵 --------
    static_path = os.path.join(RAW_DIR, "Kelmarsh_WT_static.csv")
    if os.path.exists(static_path):
        A = build_adj_from_static(static_path, turbine_ids)
    else:
        N = len(turbine_ids)
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
        "drop_jan_1_2": bool(DROP_JAN_1_2),

        "x_start": str(t_x0),
        "x_end":   str(t_x1),
        "y_start": str(t_y0),
        "y_end":   str(t_y1),

        # 年份切分：range切分 + Dataset里再做“跨年H小时”过滤（避免泄漏）
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
