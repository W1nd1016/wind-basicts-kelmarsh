# models/vector_branch_nacelle.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _auto_angle_to_rad(angle: torch.Tensor) -> torch.Tensor:
    """
    Try to infer whether angle is in degrees or radians (data-driven).
    If median(|angle|) > ~7, we treat it as degrees.
    """
    with torch.no_grad():
        a = angle.detach()
        med = torch.nanmedian(torch.abs(a))
        if torch.isfinite(med) and float(med.item()) > 7.0:
            return angle * (math.pi / 180.0)
    return angle


def angle_to_sincos(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    angle: (...), in degrees or radians (auto-detected)
    return: (sin, cos) with same shape
    """
    rad = _auto_angle_to_rad(angle)
    return torch.sin(rad), torch.cos(rad)


class VectorSetEncoder2D(nn.Module):
    """
    Encode a set of 2D vectors with (optional) neighbor position into a scalar embedding.

    v:   (B,T,N,K,2)
    pos: (N,K,3) or (B,T,N,K,3)  where 3=(dx,dy,dist) normalized
    m:   (B,T,N,K) 0/1 valid mask

    out: (B,T,N,D)
    """
    def __init__(self, out_dim: int, hidden: int = 128, use_pos: bool = True):
        super().__init__()
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.use_pos = bool(use_pos)

        in_dim = 1 + (3 if self.use_pos else 0)  # norm + pos(3)
        self.score = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

        # project pooled vector to scalar embedding
        self.to_emb = nn.Sequential(
            nn.Linear(3, self.hidden),  # vx, vy, ||v||
            nn.ReLU(),
            nn.Linear(self.hidden, self.out_dim),
        )
        self.norm = nn.LayerNorm(self.out_dim)

    def _broadcast_pos(self, pos: torch.Tensor, B: int, T: int, N: int, K: int, device) -> torch.Tensor:
        if pos.ndim == 2:
            raise RuntimeError("pos must have last dim=3, got 2D")
        if pos.ndim == 3:
            pos = pos.unsqueeze(0).unsqueeze(0)  # (1,1,N,K,3)
        elif pos.ndim == 5:
            pass
        else:
            raise RuntimeError(f"pos must be (N,K,3) or (B,T,N,K,3), got {tuple(pos.shape)}")
        pos = pos.to(device)
        if pos.size(0) == 1 and B > 1:
            pos = pos.expand(B, -1, -1, -1, -1)
        if pos.size(1) == 1 and T > 1:
            pos = pos.expand(-1, T, -1, -1, -1)
        if pos.size(2) != N or pos.size(3) != K or pos.size(4) != 3:
            raise RuntimeError(f"pos shape mismatch: got {tuple(pos.shape)}, expected (B,T,N,K,3)")
        return pos

    def forward(self, v: torch.Tensor, pos: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
        if v.ndim != 5 or v.size(-1) != 2:
            raise RuntimeError(f"v must be (B,T,N,K,2), got {tuple(v.shape)}")
        B, T, N, K, _ = v.shape

        if m is None:
            m = torch.ones((B, T, N, K), device=v.device, dtype=v.dtype)
        else:
            if m.shape != (B, T, N, K):
                raise RuntimeError(f"m must be (B,T,N,K), got {tuple(m.shape)}")
            m = m.to(device=v.device, dtype=v.dtype)

        # invariants per point
        norm = torch.sqrt((v ** 2).sum(dim=-1) + 1e-8)  # (B,T,N,K)

        if self.use_pos:
            pos5 = self._broadcast_pos(pos, B=B, T=T, N=N, K=K, device=v.device)  # (B,T,N,K,3)
            feat = torch.cat([norm.unsqueeze(-1), pos5], dim=-1)                   # (B,T,N,K,4)
        else:
            feat = norm.unsqueeze(-1)                                              # (B,T,N,K,1)

        a = self.score(feat).squeeze(-1)  # (B,T,N,K)
        a = a.masked_fill(m < 0.5, -1e9)
        w = torch.softmax(a, dim=-1).unsqueeze(-1)  # (B,T,N,K,1)

        v_pool = (w * v).sum(dim=3)                 # (B,T,N,2)
        v_norm = torch.sqrt((v_pool ** 2).sum(dim=-1) + 1e-8)  # (B,T,N)

        f = torch.cat([v_pool, v_norm.unsqueeze(-1)], dim=-1)  # (B,T,N,3)
        out = self.to_emb(f)                                   # (B,T,N,D)
        return self.norm(out)
