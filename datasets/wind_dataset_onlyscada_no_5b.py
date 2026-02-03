# datasets/wind_dataset_onlyscada_5b_match_scada_cerra_rules.py
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class WindSTFDatasetOnlyScada5B(Dataset):
    """
    X: (T3, N, F)       3小时一步（输入）
    X_valid: (T3,N,F)   0/1（用于5B筛样本 & 统计）
    Y: (T1, N)          1小时一步（监督：未来每小时功率）
    mask: (T1, N)       1小时一步（有效监督mask，含Lost过滤）

    映射： y_idx = offset + x_idx * ratio
    输出：
      x: (L, N, F)
      y: (H, N)  -> t+1..t+H
      m: (H, N)
    """
    def __init__(self, root="data/wind_onlyscada_no_5b", split="train", L=None, H=None):
        self.root = root
        self.X  = np.load(f"{root}/X.npy")
        self.Xv = np.load(f"{root}/X_valid.npy")
        self.Y  = np.load(f"{root}/Y.npy")
        self.M  = np.load(f"{root}/mask.npy")
        meta = json.load(open(f"{root}/meta.json"))

        self.L = int(meta.get("L", 9) if L is None else L)
        self.H = int(meta.get("H", 6) if H is None else H)

        self.ratio  = int(meta["ratio_x_to_y"])     # 3
        self.offset = int(meta["offset_hours"])     # 0/1/2 取决于对齐

        splits = meta["splits"]

        if split == "train":
            x0, x1 = splits["train_x"]
            y0, y1 = splits["train_y"]
            allowed_years = {2016, 2017, 2018}
        elif split == "val":
            x0, x1 = splits["val_x"]
            y0, y1 = splits["val_y"]
            allowed_years = {2019}
        else:
            x0, x1 = splits["test_x"]
            y0, y1 = splits["test_y"]
            allowed_years = {2020}

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

            # ---- split防泄漏：未来H小时最后一个点必须仍在本split年份集合里 ----
            x_time = self.x_start + x_idx * self.x_step
            y_end_time = x_time + np.timedelta64(self.H, "h")
            if int(str(y_end_time)[:4]) not in allowed_years:
                continue


            # ---- 同时要求 y_idx 落在本 split 的 y 范围内（更稳妥）----
            if (y_idx + self.H) >= self.y1 or (y_idx + 1) < self.y0:
                continue

            idxs.append(x_idx)

        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        x_idx = self.idxs[k]
        y_idx = self.offset + x_idx * self.ratio

        x = self.X[x_idx - self.L + 1 : x_idx + 1]          # (L,N,F)
        y = self.Y[y_idx + 1 : y_idx + self.H + 1]          # (H,N)
        m = self.M[y_idx + 1 : y_idx + self.H + 1]          # (H,N)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.float32),
        )
