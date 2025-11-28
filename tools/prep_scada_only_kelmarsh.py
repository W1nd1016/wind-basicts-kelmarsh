import os, re, json, glob
import numpy as np
import pandas as pd

RAW_DIR = "raw_scada"
OUT_DIR = "data/wind"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_COL = "# Date and time"
PWR_COL  = "Power (kW)"
WSP_COL  = "Wind speed (m/s)"
WDIR_COL = "Wind direction (°)"
NAC_COL  = "Nacelle position (°)"

L = 24
H = 6

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
    """
    Title: Kelmarsh 1..6
    Latitude, Longitude
    """
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

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "Turbine_Data_Kelmarsh_*.csv")))
    assert len(files)==6, f"Expect 6 turbine files, got {len(files)}"

    turb_dfs = []
    turb_ids = []

    for f in files:
        tid = turbine_id_from_filename(os.path.basename(f))
        turb_ids.append(tid)
        df = read_one_turbine_csv(f)
        df["turbine_id"] = tid
        turb_dfs.append(df)

    turb_ids = sorted(turb_ids)
    turb_dfs = [df for _,df in sorted(zip([df["turbine_id"].iloc[0] for df in turb_dfs], turb_dfs))]

    hourly_list = []
    for df in turb_dfs:
        tid = df["turbine_id"].iloc[0]
        dft = df.set_index(TIME_COL)

        h = dft.resample("1H").mean(numeric_only=True)
        h["turbine_id"] = tid
        hourly_list.append(h.reset_index())

    hdf = pd.concat(hourly_list, ignore_index=True)

    full_time = pd.date_range(hdf[TIME_COL].min(), hdf[TIME_COL].max(), freq="1H")

    P_list, W_list, Dir_list, Nac_list, M_list = [], [], [], [], []
    for tid in turb_ids:
        dft = hdf[hdf["turbine_id"]==tid].set_index(TIME_COL).reindex(full_time)
        P_list.append(dft[PWR_COL].values)
        W_list.append(dft[WSP_COL].values)
        Dir_list.append(dft[WDIR_COL].values)
        Nac_list.append(dft[NAC_COL].values)

        M_list.append(np.isfinite(dft[PWR_COL].values).astype(np.float32))

    P = np.stack(P_list, axis=1)
    W = np.stack(W_list, axis=1)
    Dir = np.stack(Dir_list, axis=1)
    Nac = np.stack(Nac_list, axis=1)
    mask = np.stack(M_list, axis=1)

    P = pd.DataFrame(P).ffill().bfill().fillna(0).values
    W = pd.DataFrame(W).ffill().bfill().fillna(0).values
    Dir = pd.DataFrame(Dir).ffill().bfill().fillna(0).values
    Nac = pd.DataFrame(Nac).ffill().bfill().fillna(0).values

    dir_rad = np.deg2rad(Dir)
    nac_rad = np.deg2rad(Nac)
    dir_sin, dir_cos = np.sin(dir_rad), np.cos(dir_rad)
    nac_sin, nac_cos = np.sin(nac_rad), np.cos(nac_rad)

    dP = np.diff(P, axis=0, prepend=P[[0],:])

    X = np.stack([P, dP, W, dir_sin, dir_cos, nac_sin, nac_cos], axis=-1).astype(np.float32)

    Y = P.astype(np.float32)

    T = X.shape[0]
    t_train = int(T*0.8)
    t_val   = int(T*0.9)

    x_mu = X[:t_train].reshape(-1, X.shape[-1]).mean(axis=0)
    x_sd = X[:t_train].reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    y_mu = Y[:t_train].mean()
    y_sd = Y[:t_train].std() + 1e-6

    Xn = (X-x_mu)/x_sd
    Yn = (Y-y_mu)/y_sd

    static_path = os.path.join(RAW_DIR, "Kelmarsh_WT_static.csv")
    if os.path.exists(static_path):
        A = build_adj_from_static(static_path, turb_ids)
    else:
        N = len(turb_ids)
        A = np.ones((N,N), dtype=np.float32)/N

    np.save(f"{OUT_DIR}/X.npy", Xn)
    np.save(f"{OUT_DIR}/Y.npy", Yn)
    np.save(f"{OUT_DIR}/mask.npy", mask.astype(np.float32))
    np.save(f"{OUT_DIR}/adj.npy", A)

    meta = {"turbine_ids": turb_ids,
            "x_mu": x_mu.tolist(),"x_sd": x_sd.tolist(),
            "y_mu": float(y_mu),"y_sd": float(y_sd),
            "splits":{"train":[0,t_train],"val":[t_train,t_val],"test":[t_val,T]},
            "feature_names":["P","dP","W","dir_sin","dir_cos","nac_sin","nac_cos"]
            }
    with open(f"{OUT_DIR}/meta.json","w") as f:
        json.dump(meta,f,indent=2)

    print("Saved to", OUT_DIR)
    print("X", Xn.shape, "Y", Yn.shape, "mask", mask.shape, "adj", A.shape)

if __name__=="__main__":
    main()