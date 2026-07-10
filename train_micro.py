# -*- coding: utf-8 -*-
"""
MicroDualNet 训练脚本 —— 使用 gen_data.py 生成的自我对弈数据训练价值头 + 策略头

用法（本地烟雾测试，几秒内跑完）：
    python train_micro.py --data selfplay_smoke.npz --epochs 2 --batch-size 8 --out micro_dual_smoke.pth

用法（Kaggle/云端正式训练，2×T4，<1小时）：
    python train_micro.py --data selfplay_data.npz --epochs 100 --batch-size 256 --out micro_dual.pth

数据格式（来自 gen_data.py 的 npz）：
    features:    (N, 60)  float32   局面特征
    boards:      (N, 10, 9) int8    局面棋盘（用于还原走法特征）
    move_scores: (N,)  object      每个局面的合法走法原始搜索评分（变长列表）
    moves_order: (N,)  object      每个局面的合法走法顺序，与 move_scores 对齐
    z_values:    (N,)  float32     局面结果标签（红方视角）

训练目标：
    loss_v = MSE(v_pred, z)                          价值头
    loss_p = MSE(scores_pred, target_scores_norm)    策略头（目标为搜索原始分数，做量级归一化后再算 MSE）
    loss   = loss_v + loss_p
"""
from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from features import FEATURE_DIM, MOVE_FEAT_DIM, _move_to_feat
from nn_evaluator import MicroDualNet
from chess_game import ChessGame

# 训练目标分数的缩放系数：搜索原始分数量级在几百~几千（子力分值），
# 缩放到与 tanh 价值头相近的量级，避免策略头 loss 数值过大主导训练。
_SCORE_SCALE = 1000.0


class SelfPlayDataset(Dataset):
    """
    每个样本：(feature_vec, move_feats(该局面全部走法), target_scores(归一化后)， z)
    走法特征在 __getitem__ 时按需计算（用 board 还原一个临时 ChessGame，不做磁盘 IO，开销很小）。
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.features = data["features"].astype(np.float32)      # (N, 60)
        self.boards = data["boards"].astype(np.int8)              # (N, 10, 9)
        self.move_scores = data["move_scores"]                    # object array of lists
        self.moves_order = data["moves_order"]                     # object array of lists
        self.z_values = data["z_values"].astype(np.float32)       # (N,)
        assert len(self.features) == len(self.z_values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        board = self.boards[idx]
        moves = self.moves_order[idx]
        scores = self.move_scores[idx]
        z = self.z_values[idx]

        g = ChessGame.__new__(ChessGame)
        g.board = [list(row) for row in board]

        move_feats = np.stack([_move_to_feat(g, tuple(m)) for m in moves], axis=0).astype(np.float32)
        target_scores = (np.asarray(scores, dtype=np.float32)) / _SCORE_SCALE

        return (
            torch.from_numpy(feat),
            torch.from_numpy(move_feats),
            torch.from_numpy(target_scores),
            torch.tensor(z, dtype=torch.float32),
        )


def collate_fn(batch):
    """变长走法数需要 padding 到该 batch 内的最大长度，并生成 mask 以在 loss 中忽略 padding。"""
    feats, move_feats_list, target_scores_list, zs = zip(*batch)

    feats = torch.stack(feats, dim=0)                     # (B, 60)
    zs = torch.stack(zs, dim=0).unsqueeze(-1)              # (B, 1)

    max_moves = max(mf.shape[0] for mf in move_feats_list)
    batch_size = len(move_feats_list)

    move_feats = torch.zeros((batch_size, max_moves, MOVE_FEAT_DIM), dtype=torch.float32)
    target_scores = torch.zeros((batch_size, max_moves), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_moves), dtype=torch.float32)

    for i, (mf, ts) in enumerate(zip(move_feats_list, target_scores_list)):
        n = mf.shape[0]
        move_feats[i, :n] = mf
        target_scores[i, :n] = ts
        mask[i, :n] = 1.0

    return feats, move_feats, target_scores, mask, zs


def _split_train_val(npz_path: str, val_ratio: float = 0.1):
    dataset = SelfPlayDataset(npz_path)
    n = len(dataset)
    n_val = max(1, int(n * val_ratio)) if n > 1 else 0
    n_train = n - n_val
    if n_val == 0:
        return dataset, dataset
    return torch.utils.data.random_split(dataset, [n_train, n_val])


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom


def main():
    parser = argparse.ArgumentParser(description="训练 MicroDualNet（价值头 + 策略头）")
    parser.add_argument("--data", type=str, default="selfplay_data.npz", help="自我对弈数据 npz 路径")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="micro_dual.pth")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 工作进程数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}, GPU 数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    print(f"加载数据: {args.data}")
    train_set, val_set = _split_train_val(args.data, args.val_ratio)
    print(f"训练集: {len(train_set)} 个局面, 验证集: {len(val_set)} 个局面")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    model = MicroDualNet(feature_dim=FEATURE_DIM, move_feat_dim=MOVE_FEAT_DIM).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"使用 DataParallel，GPU 数: {torch.cuda.device_count()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for feats, move_feats, target_scores, mask, zs in train_loader:
            feats = feats.to(device, non_blocking=True)
            move_feats = move_feats.to(device, non_blocking=True)
            target_scores = target_scores.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            zs = zs.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                v_pred, scores_pred = model(feats, move_feats)
                loss_v = nn.functional.mse_loss(v_pred, zs)
                loss_p = _masked_mse(scores_pred, target_scores, mask)
                loss = loss_v + loss_p

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            train_batches += 1

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for feats, move_feats, target_scores, mask, zs in val_loader:
                feats = feats.to(device, non_blocking=True)
                move_feats = move_feats.to(device, non_blocking=True)
                target_scores = target_scores.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                zs = zs.to(device, non_blocking=True)

                v_pred, scores_pred = model(feats, move_feats)
                loss_v = nn.functional.mse_loss(v_pred, zs)
                loss_p = _masked_mse(scores_pred, target_scores, mask)
                loss = loss_v + loss_p

                val_loss_sum += float(loss.item())
                val_batches += 1

        train_loss = train_loss_sum / max(1, train_batches)
        val_loss = val_loss_sum / max(1, val_batches)
        elapsed = time.time() - t_start
        remaining = elapsed / epoch * (args.epochs - epoch)
        print(
            f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"耗时={elapsed/60:.1f}分钟 预估剩余={remaining/60:.1f}分钟"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
            torch.save({"state_dict": state_dict}, args.out)
            print(f"  -> 验证损失改善，已保存模型到 {args.out}")

    total_time = time.time() - t_start
    print(f"训练完成，最佳验证损失 {best_val_loss:.4f}，总耗时 {total_time/60:.1f} 分钟")


if __name__ == "__main__":
    main()
