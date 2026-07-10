# -*- coding: utf-8 -*-
"""
神经网络评估模块
用于加载由 Kaggle notebook 训练并保存的局面估值模型（残差 CNN，15 通道输入）。
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict

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


# ─────────────────────────────────────────────────────────────────────────────
# MicroDualNet —— 轻量双头网络（价值头 + 策略头），用于自我对弈数据训练
# 与上方 ImprovedEvalNet（15通道 CNN，读取 model_improved.pth）完全独立、互不影响。
# ─────────────────────────────────────────────────────────────────────────────
from features import FEATURE_DIM, MOVE_FEAT_DIM, extract_features, _move_to_feat

MICRO_HIDDEN = 128
MICRO_POLICY_HIDDEN = 64


if TORCH_AVAILABLE:
    class MicroDualNet(nn.Module):
        """
        输入:
          s          形状 (B, 60)         局面特征
          move_feats 形状 (B, num_moves, 12) 走法特征（每个局面走法数可不同，调用方自行分批）
        输出:
          v      形状 (B, 1)           价值头，tanh 后 in [-1, 1]
          scores 形状 (B, num_moves)   策略头，每个候选走法一个分数（未做 softmax）
        """
        def __init__(self, feature_dim: int = FEATURE_DIM, move_feat_dim: int = MOVE_FEAT_DIM):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(feature_dim, MICRO_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Linear(MICRO_HIDDEN, MICRO_HIDDEN),
                nn.ReLU(inplace=True),
            )
            self.value_head = nn.Sequential(
                nn.Linear(MICRO_HIDDEN, 1),
                nn.Tanh(),
            )
            self.policy_head = nn.Sequential(
                nn.Linear(MICRO_HIDDEN + move_feat_dim, MICRO_POLICY_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Linear(MICRO_POLICY_HIDDEN, 1),
            )

        def forward(self, s, move_feats):
            h = self.shared(s)                      # (B, 128)
            v = self.value_head(h)                   # (B, 1)

            num_moves = move_feats.shape[1]
            h_expanded = h.unsqueeze(1).expand(-1, num_moves, -1)   # (B, num_moves, 128)
            policy_in = torch.cat([h_expanded, move_feats], dim=-1)  # (B, num_moves, 140)
            scores = self.policy_head(policy_in).squeeze(-1)         # (B, num_moves)
            return v, scores
else:
    MicroDualNet = None  # type: ignore


class NNEvaluator:
    """
    MicroDualNet 的推理封装。

    - value(game) -> float，范围 [-1, 1]，正值对 RED 有利
    - policy_scores(game, legal_moves) -> Dict[move, float]

    若 micro_dual.pth 不存在或加载失败，self.nn_available = False，
    调用方（ai_engine.py）需在该情况下回退到旧逻辑（保证向下兼容）。
    """

    def __init__(self, model_path: str = "micro_dual.pth", device: Optional[str] = None):
        self.nn_available = False
        self.model = None
        self.device = None

        if not TORCH_AVAILABLE:
            return
        import os as _os
        if not _os.path.exists(model_path):
            return

        try:
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = MicroDualNet().to(self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
            self.nn_available = True
        except Exception as e:
            print(f"Warning: failed to load MicroDualNet {model_path}: {e}")
            self.model = None
            self.device = None
            self.nn_available = False

    def value(self, game: ChessGame) -> float:
        """返回局面价值 v in [-1, 1]，正值对 RED 有利。未加载模型时抛出异常，调用方需自行判断 nn_available。"""
        if not self.nn_available:
            raise RuntimeError("MicroDualNet not available")
        s = extract_features(game)
        with torch.no_grad():
            s_t = torch.from_numpy(s).unsqueeze(0).to(self.device)       # (1, 60)
            dummy_move = torch.zeros((1, 1, MOVE_FEAT_DIM), dtype=torch.float32, device=self.device)
            v, _ = self.model(s_t, dummy_move)
            return float(v.item())

    def policy_scores(self, game: ChessGame, legal_moves) -> Dict[Tuple[int, int, int, int], float]:
        """返回每个合法走法的策略分数字典。未加载模型时抛出异常，调用方需自行判断 nn_available。"""
        if not self.nn_available:
            raise RuntimeError("MicroDualNet not available")
        if not legal_moves:
            return {}
        s = extract_features(game)
        move_feats = np.stack([_move_to_feat(game, m) for m in legal_moves], axis=0)  # (num_moves, 12)
        with torch.no_grad():
            s_t = torch.from_numpy(s).unsqueeze(0).to(self.device)                     # (1, 60)
            mf_t = torch.from_numpy(move_feats).unsqueeze(0).to(self.device)           # (1, num_moves, 12)
            _, scores = self.model(s_t, mf_t)
            scores = scores.squeeze(0).cpu().numpy()                                   # (num_moves,)
        return {m: float(sc) for m, sc in zip(legal_moves, scores)}
