# -*- coding: utf-8 -*-
"""
局面/走法特征提取模块 —— 供轻量双头网络 (MicroDualNet) 使用
不依赖任何外部文件，计算极快（纯 Python + numpy 数组运算）。

extract_features(game) -> np.ndarray, shape (60,)
    局面特征，正值总体上对红方（RED）有利。
_move_to_feat(game, move) -> np.ndarray, shape (12,)
    走法特征，需要在执行 move 之前调用（用到走子前的棋盘状态）。
"""
from __future__ import annotations
from typing import Tuple

import numpy as np

from chess_game import (
    ChessGame, RED, BLACK,
    R_KING, R_GUARD, R_BISHOP, R_ROOK, R_KNIGHT, R_CANNON, R_PAWN,
    B_KING, B_GUARD, B_BISHOP, B_ROOK, B_KNIGHT, B_CANNON, B_PAWN,
    EMPTY,
)

FEATURE_DIM = 60
MOVE_FEAT_DIM = 12

# 棋子类型顺序（用于计数、one-hot 等），索引 0~6
_PIECE_TYPES = [R_KING, R_GUARD, R_BISHOP, R_ROOK, R_KNIGHT, R_CANNON, R_PAWN]
_TYPE_MAX_COUNT = {  # 单方最大数量，用于归一化
    abs(R_KING): 1, abs(R_GUARD): 2, abs(R_BISHOP): 2,
    abs(R_ROOK): 2, abs(R_KNIGHT): 2, abs(R_CANNON): 2, abs(R_PAWN): 5,
}

# 位置分表（与 ai_engine._pst_score 独立维护，避免循环 import；
# features.py 只需要粗粒度的位置价值，不要求与手工评估表完全一致）
_ROOK_PST = [
    [14,14,12,18,16,14,12,18,14,14],
    [16,20,18,24,26,24,18,20,16, 0],
    [12,12,12,18,15,18,12,12,12, 0],
    [12,18,16,22,22,22,16,18,12, 0],
    [12,14,12,18,18,18,12,14,12, 0],
    [12,16,14,20,20,20,14,16,12, 0],
    [14,14,12,18,17,18,12,14,14, 0],
    [16,20,18,24,26,24,18,20,16, 0],
    [14,14,12,18,16,14,12,18,14, 0],
    [14,14,12,18,16,14,12,18,14,14],
]
_KNIGHT_PST = [
    [ 4, 8,16,12, 4,12,16, 8, 4],
    [ 4,10,28,16, 8,16,28,10, 4],
    [12,14,16,20,18,20,16,14,12],
    [ 8,24,18,24,20,24,18,24, 8],
    [ 6,16,14,18,16,18,14,16, 6],
    [ 4,12,16,14,12,14,16,12, 4],
    [ 4, 8, 8,12,10,12, 8, 8, 4],
    [ 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [ 0, 4, 4, 4, 4, 4, 4, 4, 0],
    [ 0, 0,10, 4, 0, 4,10, 0, 0],
]
_CANNON_PST = [
    [ 6, 4, 0,-10,-12,-10, 0, 4, 6],
    [ 2, 2, 0, -4,-14, -4, 0, 2, 2],
    [ 2, 2, 0,-10, -8,-10, 0, 2, 2],
    [ 0, 0,-2,  4, 10,  4,-2, 0, 0],
    [ 0, 0, 0,  2,  8,  2, 0, 0, 0],
    [-2, 0, 4,  2,  6,  2, 4, 0,-2],
    [ 0, 0, 0,  2,  6,  2, 0, 0, 0],
    [ 4, 0, 8,  6, 10,  6, 8, 0, 4],
    [ 0, 2, 4,  6,  6,  6, 4, 2, 0],
    [ 0, 0, 2,  6,  6,  6, 2, 0, 0],
]
_PAWN_PST_RED = [
    [ 0, 3, 6, 9,12, 9, 6, 3, 0],
    [18,36,56,80,96,80,56,36,18],
    [14,26,42,60,80,60,42,26,14],
    [10,20,30,34,40,34,30,20,10],
    [ 6,12,18,18,20,18,18,12, 6],
    [ 2, 0, 8, 0, 8, 0, 8, 0, 2],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


def _pst_score(piece: int, r: int, c: int) -> int:
    t = abs(piece)
    row = r if piece > 0 else (9 - r)
    try:
        if t == abs(R_ROOK):   return _ROOK_PST[c][row]   if piece > 0 else -_ROOK_PST[c][9-row]
        if t == abs(R_KNIGHT): return _KNIGHT_PST[row][c] if piece > 0 else -_KNIGHT_PST[9-row][c]
        if t == abs(R_CANNON): return _CANNON_PST[row][c] if piece > 0 else -_CANNON_PST[9-row][c]
        if t == abs(R_PAWN):   return _PAWN_PST_RED[row][c] if piece > 0 else -_PAWN_PST_RED[9-row][c]
    except IndexError:
        pass
    return 0


# 肋道列（0-indexed）：第 2、8 列（即棋盘从边线数第二条竖线）
_FLANK_COLS = (1, 7)
# 马入中堂：中间三列 + 中间四行范围
_CENTER_COLS = (3, 4, 5)
_CENTER_ROWS = (3, 4, 5, 6)


def extract_features(game: ChessGame) -> np.ndarray:
    """提取 60 维局面特征，正值总体上对红方有利。"""
    board = game.board
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    red_counts = {t: 0 for t in _PIECE_TYPES}
    black_counts = {t: 0 for t in _PIECE_TYPES}
    red_material = 0
    black_material = 0
    pst_sums = {abs(R_ROOK): 0, abs(R_KNIGHT): 0, abs(R_CANNON): 0, abs(R_PAWN): 0}
    red_rook_flank = 0
    black_rook_flank = 0
    red_knight_center = 0
    black_knight_center = 0
    red_rook_mob = red_cannon_mob = red_knight_mob = 0
    black_rook_mob = black_cannon_mob = black_knight_mob = 0
    red_pawn_adv = 0
    black_pawn_adv = 0
    red_king_pos = (9, 4)
    black_king_pos = (0, 4)
    material_values = {
        abs(R_GUARD): 200, abs(R_BISHOP): 200,
        abs(R_ROOK): 1000, abs(R_KNIGHT): 450, abs(R_CANNON): 450, abs(R_PAWN): 100,
    }

    for r in range(10):
        for c in range(9):
            p = board[r][c]
            if p == EMPTY:
                continue
            t = abs(p)
            is_red = p > 0
            if is_red:
                red_counts[t if p in _PIECE_TYPES else t] = red_counts.get(t, 0) + 1
            else:
                black_counts[t] = black_counts.get(t, 0) + 1

            mv = material_values.get(t, 0)
            if is_red:
                red_material += mv
            else:
                black_material += mv

            if t in pst_sums:
                pst_sums[t] += _pst_score(p, r, c)

            if t == abs(R_ROOK) and c in _FLANK_COLS:
                if is_red:
                    red_rook_flank = 1
                else:
                    black_rook_flank = 1

            if t == abs(R_KNIGHT) and c in _CENTER_COLS and r in _CENTER_ROWS:
                if is_red:
                    red_knight_center = 1
                else:
                    black_knight_center = 1

            if t == abs(R_KING):
                if is_red:
                    red_king_pos = (r, c)
                else:
                    black_king_pos = (r, c)

            if t == abs(R_PAWN):
                if is_red:
                    red_pawn_adv += (9 - r)
                else:
                    black_pawn_adv += r

    # 简单机动性估计：车/马/炮各自的行动力（复用规则生成器，成本可接受）
    saved_turn = game.current_turn
    for color, is_red in ((RED, True), (BLACK, False)):
        game.current_turn = color
        for r in range(10):
            for c in range(9):
                p = board[r][c]
                if p == EMPTY:
                    continue
                if (p > 0) != is_red:
                    continue
                t = abs(p)
                if t not in (abs(R_ROOK), abs(R_KNIGHT), abs(R_CANNON)):
                    continue
                n = len(game.get_legal_moves(r, c))
                if t == abs(R_ROOK):
                    if is_red: red_rook_mob += n
                    else: black_rook_mob += n
                elif t == abs(R_KNIGHT):
                    if is_red: red_knight_mob += n
                    else: black_knight_mob += n
                elif t == abs(R_CANNON):
                    if is_red: red_cannon_mob += n
                    else: black_cannon_mob += n
    game.current_turn = saved_turn

    idx = 0
    # Block A (14): 双方各兵种数量，按最大数量归一化
    for t in _PIECE_TYPES:
        feat[idx] = red_counts.get(t, 0) / _TYPE_MAX_COUNT[t]; idx += 1
    for t in _PIECE_TYPES:
        feat[idx] = black_counts.get(t, 0) / _TYPE_MAX_COUNT[t]; idx += 1

    # Block B (1): 子力总值差（不含帅/将）
    feat[idx] = (red_material - black_material) / 5000.0; idx += 1

    # Block C (2): 将军标志
    feat[idx] = 1.0 if game.is_in_check(RED) else 0.0; idx += 1
    feat[idx] = 1.0 if game.is_in_check(BLACK) else 0.0; idx += 1

    # Block D (4): 缺士/缺象标志（归一化为缺失比例）
    feat[idx] = (2 - red_counts.get(abs(R_GUARD), 0)) / 2.0; idx += 1
    feat[idx] = (2 - red_counts.get(abs(R_BISHOP), 0)) / 2.0; idx += 1
    feat[idx] = (2 - black_counts.get(abs(R_GUARD), 0)) / 2.0; idx += 1
    feat[idx] = (2 - black_counts.get(abs(R_BISHOP), 0)) / 2.0; idx += 1

    # Block E (2): 车占肋道
    feat[idx] = float(red_rook_flank); idx += 1
    feat[idx] = float(black_rook_flank); idx += 1

    # Block F (2): 马入中堂
    feat[idx] = float(red_knight_center); idx += 1
    feat[idx] = float(black_knight_center); idx += 1

    # Block G (4): 各棋子位置分累加值（车/马/炮/兵，已含双方符号）
    feat[idx] = pst_sums[abs(R_ROOK)] / 200.0; idx += 1
    feat[idx] = pst_sums[abs(R_KNIGHT)] / 200.0; idx += 1
    feat[idx] = pst_sums[abs(R_CANNON)] / 200.0; idx += 1
    feat[idx] = pst_sums[abs(R_PAWN)] / 200.0; idx += 1

    # Block H (1): 当前走棋方机动力（合法着法数）
    n_moves = len(game.get_all_moves(game.current_turn))
    feat[idx] = n_moves / 40.0; idx += 1

    # Block I (1): 当前走棋方标志
    feat[idx] = 1.0 if game.current_turn == RED else 0.0; idx += 1

    # Block J (2): 帅/将是否已离开原始位置
    feat[idx] = 0.0 if red_king_pos == (9, 4) else 1.0; idx += 1
    feat[idx] = 0.0 if black_king_pos == (0, 4) else 1.0; idx += 1

    # Block K (2): 子力总值（不含帅/将），绝对量
    feat[idx] = red_material / 5000.0; idx += 1
    feat[idx] = black_material / 5000.0; idx += 1

    # Block L (2): 兵/卒推进量
    feat[idx] = red_pawn_adv / 50.0; idx += 1
    feat[idx] = black_pawn_adv / 50.0; idx += 1

    # Block M (2): 车机动力
    feat[idx] = red_rook_mob / 20.0; idx += 1
    feat[idx] = black_rook_mob / 20.0; idx += 1

    # Block N (2): 炮机动力
    feat[idx] = red_cannon_mob / 20.0; idx += 1
    feat[idx] = black_cannon_mob / 20.0; idx += 1

    # Block O (2): 马机动力
    feat[idx] = red_knight_mob / 20.0; idx += 1
    feat[idx] = black_knight_mob / 20.0; idx += 1

    # Block P (1): 剩余棋子总数占比（局面阶段：开局/中局/残局）
    total_pieces = sum(red_counts.values()) + sum(black_counts.values())
    feat[idx] = total_pieces / 32.0; idx += 1

    # Block Q (4): 双方帅/将坐标（归一化）
    feat[idx] = red_king_pos[0] / 9.0; idx += 1
    feat[idx] = red_king_pos[1] / 8.0; idx += 1
    feat[idx] = black_king_pos[0] / 9.0; idx += 1
    feat[idx] = black_king_pos[1] / 8.0; idx += 1

    # 其余保留位（当前置零，供未来扩展特征使用）
    # idx 现在应等于 60 - reserved
    assert idx <= FEATURE_DIM, f"feature index overflow: {idx} > {FEATURE_DIM}"

    return feat


def _piece_type_index(piece: int) -> int:
    """将棋子值映射到 0~6 的类型索引（帅/仕/相/车/马/炮/兵）"""
    t = abs(piece)
    for i, pt in enumerate(_PIECE_TYPES):
        if abs(pt) == t:
            return i
    return 0


def _move_to_feat(game: ChessGame, move: Tuple[int, int, int, int]) -> np.ndarray:
    """
    走法特征（12维），必须在执行 move 之前调用（依赖走子前的棋盘状态）：
      [0]   fr / 9
      [1]   fc / 8
      [2]   tr / 9
      [3]   tc / 8
      [4:11] 起点棋子类型 one-hot（7维：帅/仕/相/车/马/炮/兵）
      [11]  吃子标志（目标格非空为1，否则0）
    """
    fr, fc, tr, tc = move
    feat = np.zeros(MOVE_FEAT_DIM, dtype=np.float32)
    feat[0] = fr / 9.0
    feat[1] = fc / 8.0
    feat[2] = tr / 9.0
    feat[3] = tc / 8.0
    piece = game.board[fr][fc]
    type_idx = _piece_type_index(piece)
    feat[4 + type_idx] = 1.0
    feat[11] = 1.0 if game.board[tr][tc] != EMPTY else 0.0
    return feat
