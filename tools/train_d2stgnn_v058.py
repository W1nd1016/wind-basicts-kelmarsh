import os
import sys
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

# ====== 把项目根目录加到 sys.path，方便 import datasets/models ======
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.wind_dataset import WindSTFDataset
from models.d2stgnn_v058 import D2STGNN_v058


def masked_mae(pred, true, mask=None):
    """带 mask 的 MAE；没有 mask 就退化成普通 MAE."""
    if mask is None:
        return torch.mean(torch.abs(pred - true))

    # 对齐形状，支持 [..., 1]
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if true.ndim == 4 and true.shape[-1] == 1:
        true = true.squeeze(-1)
    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    mask = (mask > 0).float()
    diff = torch.abs(pred - true) * mask
    return diff.sum() / (mask.sum() + 1e-6)


def evaluate(model, data_loader, device, epoch, name="[VAL]"):
    model.eval()
    mae_sum, rmse_sum, n_samples = 0.0, 0.0, 0

    with torch.no_grad():
        batch_seen = 0
        for batch in data_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_m = batch
            else:
                batch_x, batch_y = batch
                batch_m = None

            batch_x = batch_x.to(device)    # [B, L, N, C]
            batch_y = batch_y.to(device)    # [B, H, N] or [B, H, N, 1]
            if batch_m is not None:
                batch_m = batch_m.to(device)

            out = model(batch_x, batch_y, batch_seen, epoch, train=False)

            # 统一为 [B, H, N]
            if out.ndim == 4 and out.shape[-1] == 1:
                out_eval = out.squeeze(-1)
            else:
                out_eval = out
            if batch_y.ndim == 4 and batch_y.shape[-1] == 1:
                y_eval = batch_y.squeeze(-1)
            else:
                y_eval = batch_y

            diff = out_eval - y_eval
            mae = torch.mean(torch.abs(diff))
            rmse = torch.sqrt(torch.mean(diff ** 2))

            bsz = batch_x.size(0)
            mae_sum += mae.item() * bsz
            rmse_sum += rmse.item() * bsz
            n_samples += bsz
            batch_seen += 1

    mae_avg = mae_sum / n_samples
    rmse_avg = rmse_sum / n_samples
    print(f"{name} MAE {mae_avg:.4f} | {name} RMSE {rmse_avg:.4f}")
    return mae_avg, rmse_avg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------- 直接从 X.npy / Y.npy 里推维度 -----------------
    data_dir = os.path.join(ROOT_DIR, "data", "wind")

    X_np = np.load(os.path.join(data_dir, "X.npy"))  # [T, N, C]
    Y_np = np.load(os.path.join(data_dir, "Y.npy"))  # [T, N] 或 [T, N, 1]

    T, num_nodes, input_dim = X_np.shape
    print("X.npy shape:", X_np.shape)
    print("Y.npy shape:", Y_np.shape)
    print("num_nodes =", num_nodes)
    print("input_dim =", input_dim)

    # ==== 这里设定时间窗口 / 预测步长，需和你准备 X/Y 时一致 ====
    input_len = 24   # 过去 24 个时间步作为输入
    output_len = 6   # 预测未来 6 个时间步
    print("input_len =", input_len, "output_len =", output_len)

    # ----------------- 数据集 & DataLoader -----------------
    train_dataset = WindSTFDataset(root=data_dir, split="train",
                                   L=input_len, H=output_len)
    val_dataset   = WindSTFDataset(root=data_dir, split="val",
                                   L=input_len, H=output_len)
    test_dataset  = WindSTFDataset(root=data_dir, split="test",
                                   L=input_len, H=output_len)

    batch_size = 64

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ----------------- 读取图结构 -----------------
    adj_path = os.path.join(data_dir, "adj.npy")
    adj = np.load(adj_path)               # [N, N]
    print("Loaded adj.npy, shape:", adj.shape)
    adj = torch.tensor(adj, dtype=torch.float32)
    adjs = [adj]                          # 传 list 进去给 D2STGNN

    # ----------------- 构建 D2STGNN 模型 -----------------
    model = D2STGNN_v058(
        num_nodes=num_nodes,
        input_dim=input_dim,
        seq_len=input_len,
        horizon=output_len,
        adjs=adjs,
        k_s=3,
        k_t=3,
        node_hidden=16,
        num_hidden=64,
        time_emb_dim=16,
        gap=1,
        dropout=0.3,
    ).to(device)

    print(model.__class__.__name__, "parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ----------------- 优化器 & 损失 -----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    max_epochs = 100
    patience = 20
    best_val_mae = float("inf")
    best_epoch = 0

    batch_seen = 0

    # ----------------- 训练循环 -----------------
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_mae_sum, n_train = 0.0, 0

        for batch in train_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_m = batch
            else:
                batch_x, batch_y = batch
                batch_m = None

            batch_x = batch_x.to(device)    # [B, L, N, C]
            batch_y = batch_y.to(device)    # [B, H, N] or [B, H, N, 1]
            if batch_m is not None:
                batch_m = batch_m.to(device)

            optimizer.zero_grad()

            out = model(batch_x, batch_y, batch_seen, epoch, train=True)
            loss = masked_mae(out, batch_y, batch_m)
            loss.backward()
            optimizer.step()

            # 记录 train MAE（不用 mask 简单看趋势）
            if out.ndim == 4 and out.shape[-1] == 1:
                out_eval = out.squeeze(-1)
            else:
                out_eval = out
            if batch_y.ndim == 4 and batch_y.shape[-1] == 1:
                y_eval = batch_y.squeeze(-1)
            else:
                y_eval = batch_y

            diff = out_eval - y_eval
            mae = torch.mean(torch.abs(diff))

            bsz = batch_x.size(0)
            train_mae_sum += mae.item() * bsz
            n_train += bsz
            batch_seen += 1

        train_mae = train_mae_sum / n_train

        val_mae, val_rmse = evaluate(model, val_loader, device, epoch, name="[VAL]")

        print(
            f"[D2STGNN_v058] Epoch {epoch:03d} | "
            f"train MAE {train_mae:.4f} | "
            f"val MAE {val_mae:.4f} | val RMSE {val_rmse:.4f}"
        )

        # early stopping
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(ROOT_DIR, "output", "d2stgnn_v058_best.pt"),
            )
        elif epoch - best_epoch >= patience:
            print(
                f"Early stop at epoch {epoch}, "
                f"best epoch {best_epoch}, best val MAE {best_val_mae:.4f}"
            )
            break

    # ----------------- 测试 -----------------
    best_ckpt = os.path.join(ROOT_DIR, "output", "d2stgnn_v058_best.pt")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_mae, test_rmse = evaluate(
        model, test_loader, device, epoch=0, name="[TEST] D2STGNN_v058"
    )
    print(f"[TEST] D2STGNN_v058 MAE {test_mae:.4f} | RMSE {test_rmse:.4f}")


if __name__ == "__main__":
    main()