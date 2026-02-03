import json
import numpy as np
import torch
from torch.utils.data import Dataset

class WindSTFDatasetScadaCerra5B(Dataset):
    def __init__(self, root="data/wind_scada_cerra_5y", split="train", L=None, H=None):
        self.root = root
        self.X  = np.load(f"{root}/X.npy")
        self.Xv = np.load(f"{root}/X_valid.npy")
        self.Y  = np.load(f"{root}/Y.npy")
        self.M  = np.load(f"{root}/mask.npy")
        meta = json.load(open(f"{root}/meta.json"))

        if L is None:
            L = int(meta.get("L", 9))
        if H is None:
            H = int(meta.get("H", 6))
        self.L, self.H = L, H

        self.ratio  = int(meta["ratio_x_to_y"])
        self.offset = int(meta["offset_hours"])

        splits = meta["splits"]
        if split == "train":
            x0, x1 = splits["train_x"]
            y0, y1 = splits["train_y"]
            allowed_years = {2016, 2017}
        elif split == "val":
            x0, x1 = splits["val_x"]
            y0, y1 = splits["val_y"]
            allowed_years = {2018}
        else:
            x0, x1 = splits["test_x"]
            y0, y1 = splits["test_y"]
            allowed_years = {2019}

        self.x0, self.x1 = int(x0), int(min(x1, self.X.shape[0]))
        self.y0, self.y1 = int(y0), int(min(y1, self.Y.shape[0]))

        self.x_start = np.datetime64(meta["x_start"])
        self.x_step  = np.timedelta64(int(meta["x_step_hours"]), "h")

        idxs = []
        for x_idx in range(self.x0, self.x1):
            if x_idx < (self.L - 1):
                continue

            y_idx = self.offset + x_idx * self.ratio
            if (y_idx + self.H) >= len(self.Y):
                continue

            x_time = self.x_start + x_idx * self.x_step
            y_end_time = x_time + np.timedelta64(self.H, "h")
            if int(str(y_end_time)[:4]) not in allowed_years:
                continue

            win_v = self.Xv[x_idx - self.L + 1 : x_idx + 1]
            if not np.all(win_v > 0.5):
                continue

            m = self.M[y_idx + 1 : y_idx + self.H + 1]
            if float(m.sum()) <= 0.0:
                continue

            if (y_idx + self.H) >= self.y1 or (y_idx + 1) < self.y0:
                continue

            idxs.append(x_idx)

        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        x_idx = self.idxs[k]
        y_idx = self.offset + x_idx * self.ratio

        x = self.X[x_idx - self.L + 1 : x_idx + 1]
        y = self.Y[y_idx + 1 : y_idx + self.H + 1]
        m = self.M[y_idx + 1 : y_idx + self.H + 1]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.float32),
        )
