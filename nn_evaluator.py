# -*- coding: utf-8 -*-
"""
神经网络评估模块
用于加载由 notebook 训练并保存的局面估值模型。
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
    ChessGame, R_KING, R_GUARD, R_BISHOP, R_ROOK, R_KNIGHT, R_CANNON, R_PAWN,
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
    """将棋盘编码为 14x10x9 的张量"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for nn_evaluator")
    x = np.zeros((14, 10, 9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            p = game.board[r][c]
            if p != 0:
                idx = PIECE_TO_IDX.get(p)
                if idx is not None:
                    x[idx, r, c] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


class SmallEvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def load_nn_evaluator(path: str, device: Optional[str] = None) -> Tuple['SmallEvalNet', 'torch.device']:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load nn evaluator")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    model = SmallEvalNet().to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, device


def evaluate_board(game: ChessGame, model: 'SmallEvalNet', device: 'torch.device') -> float:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate board with nn model")
    with torch.no_grad():
        tensor = board_to_tensor(game).to(device)
        value = model(tensor)
        return float(value.item())
