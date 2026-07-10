# -*- coding: utf-8 -*-
"""
神经网络评估模块
用于加载由 Kaggle notebook 训练并保存的局面估值模型（残差 CNN，15 通道输入）。
"""
from __future__ import annotations
from typing import Optional, Tuple

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

from chess_game import (
    ChessGame, RED,
    R_KING, R_GUARD, R_BISHOP, R_ROOK, R_KNIGHT, R_CANNON, R_PAWN,
    B_KING, B_GUARD, B_BISHOP, B_ROOK, B_KNIGHT, B_CANNON, B_PAWN,
    EMPTY,
)
import numpy as np

PIECE_LIST = [
    R_KING, R_GUARD, R_BISHOP, R_ROOK, R_KNIGHT, R_CANNON, R_PAWN,
    B_KING, B_GUARD, B_BISHOP, B_ROOK, B_KNIGHT, B_CANNON, B_PAWN,
]
PIECE_TO_IDX = {p: i for i, p in enumerate(PIECE_LIST)}


def board_to_tensor(game: ChessGame) -> 'torch.Tensor':
    """
    将棋盘编码为 15x10x9 的张量：
      通道 0~13：7 种棋子 x 2 方（one-hot）
      通道 14  ：当前走棋方标志位（RED 时全 1，否则全 0）
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for nn_evaluator")
    x = np.zeros((15, 10, 9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            p = game.board[r][c]
            if p != 0:
                idx = PIECE_TO_IDX.get(p)
                if idx is not None:
                    x[idx, r, c] = 1.0
    if game.current_turn == RED:
        x[14, :, :] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


class _ResBlock(nn.Module):
    """conv.0/1/3/4 命名对应权重文件中的 tower.N.conv.* 结构"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class ImprovedEvalNet(nn.Module):
    """
    对应 model_improved.pth 的 tower.0~tower.14 结构：
      0: Conv2d(in_channels, 64, 3, padding=1)
      1: BatchNorm2d(64)
      2: ReLU
      3: ResBlock(64)
      4: ResBlock(64)
      5: Conv2d(64, 128, 3, padding=1)
      6: BatchNorm2d(128)
      7: ReLU
      8: ResBlock(128)
      9: ResBlock(128)
      10: AdaptiveAvgPool2d(1)
      11: Flatten
      12: Linear(128, 64)
      13: ReLU
      14: Linear(64, 1)
    """
    def __init__(self, in_channels: int = 15):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            _ResBlock(64),
            _ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            _ResBlock(128),
            _ResBlock(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.tower(x)


def load_nn_evaluator(path: str, device: Optional[str] = None):
    """
    加载 model_improved.pth，返回 (model, device, y_mean, y_std)。
    checkpoint 字段：state_dict / y_mean / y_std / in_channels
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load nn evaluator")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    in_channels = checkpoint.get('in_channels', 15)
    model = ImprovedEvalNet(in_channels=in_channels).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    y_mean = float(checkpoint.get('y_mean', 0.0))
    y_std = float(checkpoint.get('y_std', 1.0))
    return model, device, y_mean, y_std


def evaluate_board(game: ChessGame, model, device, y_mean: float = 0.0, y_std: float = 1.0) -> float:
    """返回局面评估分（正值 RED 有利），已做反标准化：value * y_std + y_mean"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate board with nn model")
    with torch.no_grad():
        tensor = board_to_tensor(game).to(device)
        raw = model(tensor)
        return float(raw.item()) * y_std + y_mean
