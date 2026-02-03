# datasets/wind_dataset_scada_cerra_fnp_v2.py
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class WindDatasetScadaCerraFNPv2(Dataset):
    """
    Loads:
      X        (T,N,F)
      X_valid  (T,N,F)
      Y        (T,N,H)  or (T,H,N)  (auto-handle)
      Y_valid  (T,N,H)  or (T,H,N)

    Returns per sample:
      x   : (L,N,F)
      xv  : (L,N,F)
      y   : (H,N)
      yv  : (H,N)
    """
    def __init__(self, root="data/wind_scada_cerra_fnp_v2", split="train", L=None, H=None):
        self.root = root
        self.X  = np.load(f"{root}/X.npy").astype(np.float32)
        self.Xv = np.load(f"{root}/X_valid.npy").astype(np.float32)
        self.Y  = np.load(f"{root}/Y.npy").astype(np.float32)
        self.Yv = np.load(f"{root}/Y_valid.npy").astype(np.float32)

        meta = json.load(open(f"{root}/meta.json", "r"))
        self.meta = meta

        if L is None:
            L = int(meta.get("L", 9))
        if H is None:
            H = int(meta.get("H", 6))
        self.L, self.H = int(L), int(H)

        T = self.X.shape[0]
        if self.Y.shape[0] != T:
            raise RuntimeError(f"Time length mismatch: X.T={T} but Y.T={self.Y.shape[0]}")
        if self.Xv.shape != self.X.shape:
            raise RuntimeError(f"X_valid shape mismatch: {self.Xv.shape} vs {self.X.shape}")

        # ---- split indices (robust) ----
        splits = meta.get("splits", {})
        # supports:
        #  A) splits={"train":[a,b], "val":[c,d], "test":[e,f]}
        #  B) splits={"train_x":[a,b], "val_x":[c,d], "test_x":[e,f]}
        if split in splits and isinstance(splits[split], (list, tuple)) and len(splits[split]) == 2:
            s0, s1 = int(splits[split][0]), int(splits[split][1])
        elif f"{split}_x" in splits and isinstance(splits[f"{split}_x"], (list, tuple)) and len(splits[f"{split}_x"]) == 2:
            s0, s1 = int(splits[f"{split}_x"][0]), int(splits[f"{split}_x"][1])
        else:
            # fallback: use whole range
            s0, s1 = 0, T

        s0 = max(0, min(s0, T))
        s1 = max(0, min(s1, T))
        if s1 <= s0:
            raise RuntimeError(f"Bad split range: {split} => [{s0},{s1})")

        idxs = []
        for t in range(s0, s1):
            if t < self.L - 1:
                continue
            # require at least one valid target step (or you can require all valid)
            y, yv = self._get_y(t)
            if float(yv.sum()) <= 0.0:
                continue
            idxs.append(t)

        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def _get_y(self, t: int):
        y = self.Y[t]
        yv = self.Yv[t]

        # handle (N,H) vs (H,N)
        if y.ndim != 2:
            raise RuntimeError(f"Y[t] must be 2D, got {y.shape}")
        H = self.H

        if y.shape[0] == H:
            # (H,N)
            y_hn = y
            yv_hn = yv
        elif y.shape[1] == H:
            # (N,H) -> (H,N)
            y_hn = y.T
            yv_hn = yv.T
        else:
            raise RuntimeError(f"Cannot infer Y layout: Y[t].shape={y.shape}, expected H={H} in one dim")

        return y_hn.astype(np.float32), yv_hn.astype(np.float32)

    def __getitem__(self, k):
        t = self.idxs[k]
        x  = self.X [t - self.L + 1 : t + 1]     # (L,N,F)
        xv = self.Xv[t - self.L + 1 : t + 1]     # (L,N,F)

        y, yv = self._get_y(t)                   # (H,N)

        return (
            torch.tensor(x,  dtype=torch.float32),
            torch.tensor(xv, dtype=torch.float32),
            torch.tensor(y,  dtype=torch.float32),
            torch.tensor(yv, dtype=torch.float32),
        )
