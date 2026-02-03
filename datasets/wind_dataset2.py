import json
import numpy as np
import torch
from torch.utils.data import Dataset

class WindSTFDataset2(Dataset):
    """
    多时间尺度数据集：
      X:    (T3, N, F)   3小时一步（输入特征）
      Y:    (T1, N)      1小时一步（监督标签：未来每小时功率）
      mask: (T1, N)      1小时一步（有效标签mask：已包含限电/停机过滤规则）

    每个样本锚点是一个3小时刻 x_idx 对应的时间 t：
      输入:  X[x_idx-L+1 : x_idx+1]            -> (L, N, F)
      输出:  Y[y_idx+1 : y_idx+H+1]            -> (H, N)  (每小时)
             mask 同理

    映射关系：
      y_idx = offset_hours + x_idx * ratio
    其中 ratio=3 (3h->1h), offset_hours 用于对齐 x_start 与 y_start 的偏移。
    """
    def __init__(self, root="data/wind4_1hour", split="train", L=None, H=None):
        self.root = root
        self.X = np.load(f"{root}/X.npy")
        self.Y = np.load(f"{root}/Y.npy")
        self.M = np.load(f"{root}/mask.npy")
        meta = json.load(open(f"{root}/meta.json"))

        if L is None:
            L = int(meta.get("L", 9))
        if H is None:
            H = int(meta.get("H", 6))
        self.L, self.H = L, H

        self.ratio = int(meta["ratio_x_to_y"])
        self.offset = int(meta["offset_hours"])

        T3 = self.X.shape[0]
        splits = meta["splits"]
        tr0, tr1 = splits["train_x"]
        va0, va1 = splits["val_x"]
        te0, te1 = splits["test_x"]

        if split == "train":
            x0, x1 = tr0, tr1
        elif split == "val":
            x0, x1 = va0, va1
        else:
            x0, x1 = te0, te1
        idxs = []
        for x_idx in range(x0, x1):
            if x_idx < (self.L - 1):
                continue
            y_idx = self.offset + x_idx * self.ratio
            if (y_idx + self.H) < len(self.Y):
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
