import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.agcrn_seq2seq_baseline import AGCRNSeq2SeqBaseline  # 用你工程里现有实现


def _try_load_npz(path: str):
    z = np.load(path)
    keys = list(z.keys())
    # allow common key variants
    if ("dx" in z) and ("dy" in z) and ("dist" in z):
        return z["dx"], z["dy"], z["dist"], keys
    if ("dx" in z) and ("dy" in z) and ("ds" in z):
        return z["dx"], z["dy"], z["ds"], keys
    if ("dx_m" in z) and ("dy_m" in z) and ("dist_m" in z):
        return z["dx_m"], z["dy_m"], z["dist_m"], keys
    if ("dx_m" in z) and ("dy_m" in z) and ("ds_m" in z):
        return z["dx_m"], z["dy_m"], z["ds_m"], keys
    return None, None, None, keys


def _ensure_NK(a: np.ndarray, N: int):
    if a.ndim != 2:
        return None
    if a.shape[0] == N:
        return a
    if a.shape[1] == N:
        return a.T
    return None


def _load_neighbors_auto(root_dir: str, N: int, preferred_prefix: str = None):
    """
    Tries in order:
      1) preferred_prefix_* files (dx/dy/dist) if provided
      2) any neighbors*.npz in root_dir
      3) any triplet of *dx*.npy/*dy*.npy/*dist|ds*.npy that contains 'neighbor'
      4) any single neighbors*.npy with shape (N,K,3) or (N,3,K) or (3,N,K)

    Returns: dx, dy, ds (all float32), and a debug string.
    """
    cand_logs = []

    def log(s):
        cand_logs.append(s)

    # ---------- (1) preferred prefix ----------
    if preferred_prefix is not None:
        base = preferred_prefix
        trials = [
            (f"{base}_dx.npy", f"{base}_dy.npy", f"{base}_dist.npy"),
            (f"{base}_dx.npy", f"{base}_dy.npy", f"{base}_ds.npy"),
            (f"{base}_dx.npy", f"{base}_dy.npy", f"{base}_distance.npy"),
        ]
        for dxp, dyp, dsp in trials:
            if os.path.exists(dxp) and os.path.exists(dyp) and os.path.exists(dsp):
                dx = _ensure_NK(np.load(dxp).astype(np.float32), N)
                dy = _ensure_NK(np.load(dyp).astype(np.float32), N)
                ds = _ensure_NK(np.load(dsp).astype(np.float32), N)
                if dx is not None and dy is not None and ds is not None:
                    log(f"[NEIGHBORS] loaded triplet: {dxp}, {dyp}, {dsp}")
                    return dx, dy, ds, "\n".join(cand_logs)

        npz_path = f"{base}.npz"
        if os.path.exists(npz_path):
            dx0, dy0, ds0, keys = _try_load_npz(npz_path)
            if dx0 is not None:
                dx = _ensure_NK(dx0.astype(np.float32), N)
                dy = _ensure_NK(dy0.astype(np.float32), N)
                ds = _ensure_NK(ds0.astype(np.float32), N)
                if dx is not None and dy is not None and ds is not None:
                    log(f"[NEIGHBORS] loaded npz: {npz_path} keys={keys}")
                    return dx, dy, ds, "\n".join(cand_logs)
            log(f"[NEIGHBORS] found npz but keys not matched: {npz_path} keys={keys}")

        npy_path = f"{base}.npy"
        if os.path.exists(npy_path):
            arr = np.load(npy_path).astype(np.float32)
            # try packed shapes
            dx, dy, ds = _unpack_neighbors_single(arr, N)
            if dx is not None:
                log(f"[NEIGHBORS] loaded packed npy: {npy_path} shape={arr.shape}")
                return dx, dy, ds, "\n".join(cand_logs)

    # ---------- (2) scan any npz ----------
    npz_list = sorted(glob.glob(os.path.join(root_dir, "*neighbor*.npz")) + glob.glob(os.path.join(root_dir, "*neighbors*.npz")))
    for p in npz_list:
        dx0, dy0, ds0, keys = _try_load_npz(p)
        if dx0 is None:
            log(f"[NEIGHBORS] skip npz(no keys): {os.path.basename(p)} keys={keys}")
            continue
        dx = _ensure_NK(dx0.astype(np.float32), N)
        dy = _ensure_NK(dy0.astype(np.float32), N)
        ds = _ensure_NK(ds0.astype(np.float32), N)
        if dx is not None and dy is not None and ds is not None:
            log(f"[NEIGHBORS] loaded npz(auto): {os.path.basename(p)} keys={keys}")
            return dx, dy, ds, "\n".join(cand_logs)
        log(f"[NEIGHBORS] skip npz(shape mismatch): {os.path.basename(p)} dx={dx0.shape} dy={dy0.shape} ds={ds0.shape}")

    # ---------- (3) scan triplet npy ----------
    all_npy = sorted(glob.glob(os.path.join(root_dir, "*.npy")))
    neigh_npy = [p for p in all_npy if ("neigh" in os.path.basename(p).lower())]

    def pick(patterns):
        out = []
        for p in neigh_npy:
            bn = os.path.basename(p).lower()
            if any(pt in bn for pt in patterns):
                out.append(p)
        return sorted(out)

    dx_cands = pick(["dx"])
    dy_cands = pick(["dy"])
    ds_cands = pick(["dist", "ds", "distance"])

    # brute force best match by trying combos
    for dxp in dx_cands:
        for dyp in dy_cands:
            for dsp in ds_cands:
                try:
                    dx0 = np.load(dxp).astype(np.float32)
                    dy0 = np.load(dyp).astype(np.float32)
                    ds0 = np.load(dsp).astype(np.float32)
                except Exception as e:
                    log(f"[NEIGHBORS] load error triplet: {os.path.basename(dxp)},{os.path.basename(dyp)},{os.path.basename(dsp)} err={e}")
                    continue
                dx = _ensure_NK(dx0, N)
                dy = _ensure_NK(dy0, N)
                ds = _ensure_NK(ds0, N)
                if dx is not None and dy is not None and ds is not None:
                    log(f"[NEIGHBORS] loaded triplet(auto): {os.path.basename(dxp)}, {os.path.basename(dyp)}, {os.path.basename(dsp)}")
                    return dx, dy, ds, "\n".join(cand_logs)

    # ---------- (4) scan single packed neighbors npy ----------
    packed = [p for p in neigh_npy if ("dx" not in os.path.basename(p).lower() and "dy" not in os.path.basename(p).lower() and "dist" not in os.path.basename(p).lower() and "ds" not in os.path.basename(p).lower())]
    for p in packed:
        try:
            arr = np.load(p).astype(np.float32)
        except Exception as e:
            log(f"[NEIGHBORS] load error packed: {os.path.basename(p)} err={e}")
            continue
        dx, dy, ds = _unpack_neighbors_single(arr, N)
        if dx is not None:
            log(f"[NEIGHBORS] loaded packed(auto): {os.path.basename(p)} shape={arr.shape}")
            return dx, dy, ds, "\n".join(cand_logs)

    # fail: list relevant files
    listing = sorted([os.path.basename(p) for p in neigh_npy] + [os.path.basename(p) for p in npz_list])
    raise RuntimeError(
        "Cannot find neighbors files in root_dir.\n"
        f"root_dir={root_dir}\n"
        f"preferred_prefix={preferred_prefix}\n"
        f"matched_files={listing}\n"
        "Logs:\n" + "\n".join(cand_logs)
    )


def _unpack_neighbors_single(arr: np.ndarray, N: int):
    """
    Accepts packed neighbor arrays:
      (N,K,3) -> dx,dy,dist
      (N,3,K) -> dx,dy,dist
      (3,N,K) -> dx,dy,dist
    """
    if arr.ndim != 3:
        return None, None, None
    a = arr
    # (N,K,3)
    if a.shape[0] == N and a.shape[2] == 3:
        dx = a[:, :, 0]
        dy = a[:, :, 1]
        ds = a[:, :, 2]
        return dx.astype(np.float32), dy.astype(np.float32), ds.astype(np.float32)
    # (N,3,K)
    if a.shape[0] == N and a.shape[1] == 3:
        dx = a[:, 0, :]
        dy = a[:, 1, :]
        ds = a[:, 2, :]
        return dx.astype(np.float32), dy.astype(np.float32), ds.astype(np.float32)
    # (3,N,K)
    if a.shape[0] == 3 and a.shape[1] == N:
        dx = a[0, :, :]
        dy = a[1, :, :]
        ds = a[2, :, :]
        return dx.astype(np.float32), dy.astype(np.float32), ds.astype(np.float32)
    return None, None, None


class SetConv(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.proj = nn.Linear(din, dout)
        self.log_lt = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, coords):
        B, P, _ = x.shape
        h = self.proj(x)

        t = coords[:, 0:1]
        xy = coords[:, 1:3]

        dt = t - t.t()
        dxy = xy[:, None, :] - xy[None, :, :]

        lt = torch.exp(self.log_lt) + 1e-6
        ls = torch.exp(self.log_ls) + 1e-6

        dist2 = (dt / lt) ** 2 + (dxy[..., 0] / ls) ** 2 + (dxy[..., 1] / ls) ** 2
        K = torch.exp(-0.5 * dist2).to(x.device)
        W = K / (K.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.einsum("pq,bqd->bpd", W, h)
        return out


class NeuralFourierLayer(nn.Module):
    def __init__(self, d_model, modes=8, kernel_size=3):
        super().__init__()
        self.d = d_model
        self.modes = modes
        self.Wr = nn.Parameter(torch.randn(modes, d_model, d_model) * 0.02)
        self.Wi = nn.Parameter(torch.randn(modes, d_model, d_model) * 0.02)
        pad = kernel_size // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=pad, groups=1)
        self.norm = nn.LayerNorm(d_model)

    def _complex_linear(self, x_fft):
        xr = x_fft.real
        xi = x_fft.imag
        K = x_fft.size(2)

        Wr = self.Wr[:K]
        Wi = self.Wi[:K]

        yr = torch.einsum("bnkd,kdf->bnkf", xr, Wr) - torch.einsum("bnkd,kdf->bnkf", xi, Wi)
        yi = torch.einsum("bnkd,kdf->bnkf", xr, Wi) + torch.einsum("bnkd,kdf->bnkf", xi, Wr)
        return torch.complex(yr, yi)

    def forward(self, x):
        B, N, L, D = x.shape
        x_fft = torch.fft.rfft(x, dim=2)
        Lf = x_fft.shape[2]
        K = min(self.modes, Lf)

        y_fft = x_fft.clone()
        y_fft[:, :, :K, :] = self._complex_linear(x_fft[:, :, :K, :]) + x_fft[:, :, :K, :]
        y = torch.fft.irfft(y_fft, n=L, dim=2)

        y2 = x.permute(0, 1, 3, 2).reshape(B * N, D, L)
        y2 = self.conv(y2).reshape(B, N, D, L).permute(0, 1, 3, 2)

        out = self.norm(y + y2)
        return out


class FuncRepVFR(nn.Module):
    def __init__(self, in_dim, d_model, modes=8):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.setconv = SetConv(d_model, d_model)
        self.nfl = NeuralFourierLayer(d_model, modes=modes)

    def forward(self, x, coords):
        B, L, N, _ = x.shape
        h = self.in_proj(x)
        h = h.reshape(B, L * N, -1)
        h = self.setconv(h, coords)
        h = h.reshape(B, L, N, -1).permute(0, 2, 1, 3)
        h = self.nfl(h)
        h = h.permute(0, 2, 1, 3)
        return h


class GridPool(nn.Module):
    def __init__(self, v_in: int, d_model: int):
        super().__init__()
        self.v_in = int(v_in)
        self.d_model = int(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.v_in + 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.dist_scale = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats, dx, dy, ds, v_point=None):
        pos = torch.cat([dx, dy, ds], dim=-1)
        p = self.mlp(torch.cat([feats, pos], dim=-1))
        a = self.attn(p).squeeze(-1)
        a = a - F.softplus(self.dist_scale) * ds.squeeze(-1)
        if v_point is not None:
            a = a.masked_fill(v_point < 0.5, -1e9)
        w = torch.softmax(a, dim=-1).unsqueeze(-1)
        z = (w * p).sum(dim=-2)
        return self.norm(z)


class DAMSoft(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.shared = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.smooth = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, z_bg, z_obs):
        y_dot = 0.5 * (self.shared(z_bg) + self.shared(z_obs))
        g_in = torch.cat([z_obs, z_bg, y_dot], dim=-1)
        alpha = torch.sigmoid(self.gate(g_in))
        y_sel = alpha * z_obs + (1.0 - alpha) * z_bg
        z = self.fuse(torch.cat([y_dot, y_sel], dim=-1))

        B, L, N, D = z.shape
        z2 = z.permute(0, 2, 3, 1).reshape(B * N, D, L)
        z2 = self.smooth(z2).reshape(B, N, D, L).permute(0, 3, 1, 2)
        return z + z2


class FNPFusionV3(nn.Module):
    def __init__(self, obs_dim, d_model, L, H, K, blocks, vars_per_point, modes=8):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.d_model = int(d_model)
        self.L = int(L)
        self.H = int(H)
        self.K = int(K)
        self.blocks = int(blocks)
        self.vars_per_point = int(vars_per_point)

        self.obs_enc = FuncRepVFR(obs_dim, d_model, modes=modes)
        self.grid_pool = GridPool(v_in=self.vars_per_point, d_model=d_model)
        self.dam = DAMSoft(d_model)

    def _reshape_bg(self, x_bg):
        B, L, N, D = x_bg.shape
        K = self.K
        V = self.vars_per_point
        blocks = self.blocks
        expect = blocks * K * V
        if D != expect:
            raise RuntimeError(f"bg dim mismatch: got {D}, expect {expect} (blocks={blocks},K={K},V={V})")
        return x_bg.view(B, L, N, blocks, K, V)

    def forward(self, x_obs, x_bg, coords, dx, dy, ds, x_bg_valid=None):
        z_obs = self.obs_enc(x_obs, coords)

        bg = self._reshape_bg(x_bg)
        bg_an = bg[:, :, :, 0, :, :]

        if x_bg_valid is not None:
            bgv = self._reshape_bg(x_bg_valid)
            v_an = (bgv[:, :, :, 0, :, :].mean(dim=-1) > 0.5).float()
        else:
            v_an = None

        z_bg = self.grid_pool(
            bg_an,
            dx.expand(x_bg.shape[0], self.L, -1, -1, -1),
            dy.expand(x_bg.shape[0], self.L, -1, -1, -1),
            ds.expand(x_bg.shape[0], self.L, -1, -1, -1),
            v_point=v_an,
        )

        z_hist = self.dam(z_bg, z_obs)

        bg_last = bg[:, -1]
        if self.blocks < 1 + self.H:
            raise RuntimeError(f"blocks({self.blocks}) < 1+H({1+self.H})")
        bg_fc = bg_last[:, :, 1 : 1 + self.H, :, :]
        bg_fc = bg_fc.permute(0, 2, 1, 3, 4).contiguous()

        if x_bg_valid is not None:
            bgv_last = self._reshape_bg(x_bg_valid)[:, -1]
            v_fc = bgv_last[:, :, 1 : 1 + self.H, :, :].mean(dim=-1)
            v_fc = (v_fc > 0.5).float().permute(0, 2, 1, 3).contiguous()
        else:
            v_fc = None

        z_bg_future = self.grid_pool(
            bg_fc,
            dx.expand(x_bg.shape[0], self.H, -1, -1, -1),
            dy.expand(x_bg.shape[0], self.H, -1, -1, -1),
            ds.expand(x_bg.shape[0], self.H, -1, -1, -1),
            v_point=v_fc,
        )

        return z_hist, z_bg_future, v_fc


class FutureResidualHead(nn.Module):
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_bg_future):
        return self.mlp(z_bg_future).squeeze(-1)


class FNP_AGCRN_ModelV3(nn.Module):
    def __init__(
        self,
        root_meta_path: str,
        neighbors_path_prefix: str = None,
        obs_dim: int = 7,
        d_model: int = 128,
        agcrn_hidden: int = 64,
        agcrn_embed: int = 10,
        agcrn_K: int = 2,
        dropout: float = 0.1,
        modes: int = 8,
    ):
        super().__init__()
        meta = json.load(open(root_meta_path, "r"))
        self.meta = meta

        self.obs_dim = int(obs_dim)
        self.L = int(meta.get("L", 9))
        self.H = int(meta.get("H", 6))
        self.N = len(meta.get("turbine_ids", [])) or int(meta.get("N", 6))
        self.blocks = 1 + self.H
        self.d_model = int(d_model)
        self.modes = int(modes)

        xy = np.array(meta["turbine_xy"], dtype=np.float32)
        if xy.shape[0] != self.N:
            raise RuntimeError(f"turbine_xy N mismatch: {xy.shape[0]} vs meta N={self.N}")

        t = np.linspace(-1.0, 1.0, self.L, dtype=np.float32)[:, None]
        coords = []
        for i in range(self.L):
            ti = np.repeat(t[i : i + 1], self.N, axis=0)
            coords.append(np.concatenate([ti, xy], axis=1))
        coords = np.concatenate(coords, axis=0)
        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32), persistent=False)

        root_dir = os.path.dirname(root_meta_path)
        pref = None
        if neighbors_path_prefix is not None:
            # allow passing either "data/.../neighbors" or None
            pref = neighbors_path_prefix

        dx, dy, ds, debug = _load_neighbors_auto(root_dir=root_dir, N=self.N, preferred_prefix=pref)
        # normalize
        dx = dx / (np.max(np.abs(dx)) + 1e-6)
        dy = dy / (np.max(np.abs(dy)) + 1e-6)
        ds = ds / (np.max(ds) + 1e-6)

        K = dx.shape[1]
        self.K = int(K)

        dx_t = torch.tensor(dx, dtype=torch.float32).view(1, 1, self.N, self.K, 1)
        dy_t = torch.tensor(dy, dtype=torch.float32).view(1, 1, self.N, self.K, 1)
        ds_t = torch.tensor(ds, dtype=torch.float32).view(1, 1, self.N, self.K, 1)
        self.register_buffer("dx", dx_t, persistent=False)
        self.register_buffer("dy", dy_t, persistent=False)
        self.register_buffer("ds", ds_t, persistent=False)

        # 打印你要的“别猜”的信息：到底加载了哪个 neighbors 文件
        print("[NEIGHBORS][AUTO]\n" + debug)
        print(f"[NEIGHBORS] dx/dy/dist shapes: {dx.shape} {dy.shape} {ds.shape}  => K={self.K}")

        self._built = False
        self._f_total = None

        self.agcrn = AGCRNSeq2SeqBaseline(
            num_nodes=self.N,
            input_dim=self.d_model,
            hidden_dim=int(agcrn_hidden),
            embed_dim=int(agcrn_embed),
            horizon=self.H,
            K=int(agcrn_K),
            topk=None,  # 如果你版本不吃 topk，就去掉这一行参数
            dropout=float(dropout),
        )

        self.future_head = FutureResidualHead(d_model=self.d_model, hidden=64)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _maybe_build(self, F_total: int):
        if self._built and self._f_total == int(F_total):
            return

        self._f_total = int(F_total)
        Fbg = F_total - self.obs_dim
        if Fbg <= 0:
            raise RuntimeError(f"F_total={F_total} obs_dim={self.obs_dim} => bg_dim<=0")

        denom = self.blocks * self.K
        if Fbg % denom != 0:
            raise RuntimeError(
                f"Cannot infer vars_per_point: bg_dim={Fbg} not divisible by blocks*K={denom} "
                f"(blocks={self.blocks},K={self.K})"
            )
        vars_per_point = Fbg // denom
        self.vars_per_point = int(vars_per_point)

        self.fnp = FNPFusionV3(
            obs_dim=self.obs_dim,
            d_model=self.d_model,
            L=self.L,
            H=self.H,
            K=self.K,
            blocks=self.blocks,
            vars_per_point=self.vars_per_point,
            modes=self.modes,
        )
        self._built = True

        print(f"[BG] inferred blocks={self.blocks} K={self.K} vars_per_point={self.vars_per_point} bg_dim={Fbg}")

    def forward(self, x, xv=None, teacher_forcing_y=None, teacher_forcing_ratio: float = 0.0):
        B, L, N, F = x.shape
        if L != self.L:
            raise RuntimeError(f"L mismatch: got {L}, expect {self.L}")
        if N != self.N:
            raise RuntimeError(f"N mismatch: got {N}, expect {self.N}")

        self._maybe_build(F)

        x_obs = x[..., : self.obs_dim]
        x_bg = x[..., self.obs_dim :]

        x_bg_v = None
        if xv is not None:
            x_bg_v = xv[..., self.obs_dim :]

        z_hist, z_bg_future, v_fc = self.fnp(
            x_obs=x_obs,
            x_bg=x_bg,
            coords=self.coords,
            dx=self.dx,
            dy=self.dy,
            ds=self.ds,
            x_bg_valid=x_bg_v,
        )

        if teacher_forcing_y is None:
            base = self.agcrn(z_hist)
        else:
            base = self.agcrn(z_hist, teacher_forcing_y=teacher_forcing_y, teacher_forcing_ratio=teacher_forcing_ratio)

        delta = self.future_head(z_bg_future)

        if v_fc is not None:
            gate = (v_fc.mean(dim=-1) > 0.5).float()
            delta = delta * gate

        return base + self.gamma * delta
