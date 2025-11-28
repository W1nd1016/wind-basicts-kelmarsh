import numpy as np
import torch
from torch.utils.data import Dataset

class WindSTFDataset(Dataset):
    def __init__(self, root="data/wind", split="train", L=24, H=6):
        X = np.load(f"{root}/X.npy")
        Y = np.load(f"{root}/Y.npy")
        M = np.load(f"{root}/mask.npy")

        T = X.shape[0]
        t_train = int(T*0.8)
        t_val   = int(T*0.9)

        if split=="train": sl=slice(0,t_train)
        elif split=="val": sl=slice(t_train,t_val)
        else: sl=slice(t_val,T)

        self.X, self.Y, self.M = X[sl], Y[sl], M[sl]
        self.L, self.H = L, H
        self.idxs = list(range(L-1, len(self.X)-H))

    def __len__(self): return len(self.idxs)

    def __getitem__(self, k):
        t = self.idxs[k]
        x = self.X[t-self.L+1:t+1]
        y = self.Y[t+1:t+self.H+1]
        m = self.M[t+1:t+self.H+1]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(m, dtype=torch.float32))
