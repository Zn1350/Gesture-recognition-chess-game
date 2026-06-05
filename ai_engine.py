# -*- coding: utf-8 -*-
"""
智能博弈模块 —— 自适应放水 AI
核心目标：最大化玩家胜率至 70%~90%，AI 棋力严格匹配用户实时棋力分

放水策略（基于 skill_score 0~100）：
  ┌──────────┬──────────┬──────────┬───────────────────────────────┐
  │ 棋力分   │ 搜索深度 │ 放水概率 │ 放水行为                      │
  ├──────────┼──────────┼──────────┼───────────────────────────────┤
  │  0~20    │    1     │  0.70    │ 随机走子 + 主动送子           │
  │ 20~40    │    1     │  0.55    │ 随机走子 + 偶尔漏防           │
  │ 40~60    │    2     │  0.35    │ 选次优解 + 轻度漏防           │
  │ 60~80    │    2     │  0.18    │ 偶尔选次优解                  │
  │ 80~100   │    3     │  0.05    │ 接近全力，仅极少失误          │
  └──────────┴──────────┴──────────┴───────────────────────────────┘

放水动作分级（BLUNDER_MODE）：
  RANDOM    - 随机选一个合法走法（最强让棋）
  GIVEAWAY  - 主动送出高价值子（车/马/炮）
  SUBOPTIMAL- 从评分最差的前 N 个走法中随机选
  MINOR     - 从评分中间段走法中随机选
"""

from __future__ import annotations
import random
from typing import List, Optional, Tuple

from chess_game import (
    ChessGame, RED, BLACK,
    R_KING, R_GUARD, R_BISHOP, R_ROOK, R_KNIGHT, R_CANNON, R_PAWN,
    B_KING, B_GUARD, B_BISHOP, B_ROOK, B_KNIGHT, B_CANNON, B_PAWN,
    PIECE_VALUES, EMPTY,
)

# ── 位置价值表 ────────────────────────────────────────────────────────────────
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

# 高价值子（主动送子时优先送这些）
_HIGH_VALUE_PIECES = {B_ROOK, B_KNIGHT, B_CANNON}


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


def _score_to_ai_params(skill_score: float) -> Tuple[int, float]:
    """
    根据实时棋力分计算 AI 参数
    返回 (search_depth, blunder_rate)

    # 棋力分越高 → 搜索深度越深，放水概率越低
    # 目标胜率 70%~90%：blunder_rate 为主要调节旋钮
    """
    if skill_score < 20:
        return 1, 0.70   # 棋力分 0~20：深度1，放水70%
    if skill_score < 40:
        return 1, 0.55   # 棋力分 20~40：深度1，放水55%
    if skill_score < 60:
        return 2, 0.35   # 棋力分 40~60：深度2，放水35%
    if skill_score < 80:
        return 2, 0.18   # 棋力分 60~80：深度2，放水18%
    return 3, 0.05       # 棋力分 80~100：深度3，放水5%


class AIEngine:
    """
    自适应放水 AI 引擎

    每次 get_move() 接收实时 skill_score，动态决定：
      1. 是否触发放水（概率 = blunder_rate）
      2. 放水级别（RANDOM / GIVEAWAY / SUBOPTIMAL / MINOR）
      3. 正常搜索深度（search_depth）

    将 ai_blunder_rate / ai_search_depth 回写到 stats，供 HUD 显示
    """

    def __init__(self):
        self._nodes = 0

    def get_move(
        self,
        game: ChessGame,
        skill_score: float,
        stats=None,          # RehabStats，用于回写 AI 参数供 HUD 显示
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        返回 AI 选择的走法 (fr, fc, tr, tc)

        # 实时棋力分: skill_score
        # AI放水率:   blunder_rate（见 _score_to_ai_params）
        # 强化学习奖励值: 由 AssessmentModule 在 main.py 中记录
        """
        moves = game.get_all_moves(BLACK)
        if not moves:
            return None

        depth, blunder_rate = _score_to_ai_params(skill_score)

        # 回写 AI 参数到 stats，HUD 可展示
        if stats is not None:
            stats.ai_blunder_rate = blunder_rate   # 注释：当前AI放水率
            stats.ai_search_depth = depth          # 注释：当前AI搜索深度

        # ── 放水决策 ──────────────────────────────────────────────────────────
        if random.random() < blunder_rate:
            return self._blunder_move(game, moves, skill_score)

        # ── 正常 Alpha-Beta 搜索 ──────────────────────────────────────────────
        return self._best_move(game, moves, depth, skill_score)

    # ── 放水走法生成 ──────────────────────────────────────────────────────────

    def _blunder_move(
        self,
        game: ChessGame,
        moves: List[Tuple],
        skill_score: float,
    ) -> Tuple[int, int, int, int]:
        """
        分级放水：根据棋力分选择放水力度

        # skill_score < 20：GIVEAWAY（主动送高价值子）
        # skill_score < 40：RANDOM（完全随机走子）
        # skill_score < 60：SUBOPTIMAL（最差的前1/3走法）
        # skill_score >= 60：MINOR（中间段走法）
        """
        if skill_score < 20:
            # GIVEAWAY：尝试主动把高价值子走到对方能吃的位置
            giveaway = self._find_giveaway_move(game, moves)
            if giveaway:
                return giveaway
            return random.choice(moves)

        if skill_score < 40:
            # RANDOM：完全随机，制造明显失误
            return random.choice(moves)

        # 先对所有走法评分
        scored = self._score_all_moves(game, moves)
        # scored 已按 score 升序（越小对 BLACK 越好）

        if skill_score < 60:
            # SUBOPTIMAL：从最差的前 1/3 走法中随机选
            worst_n = max(1, len(scored) // 3)
            return random.choice(scored[-worst_n:])[1]

        # MINOR（60~80）：从中间段走法中随机选，避免最优也避免最差
        n = len(scored)
        mid_start = n // 4
        mid_end   = max(mid_start + 1, n * 3 // 4)
        return random.choice(scored[mid_start:mid_end])[1]

    def _find_giveaway_move(
        self,
        game: ChessGame,
        moves: List[Tuple],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        主动送子：找到能让红方（玩家）下一步吃掉 AI 高价值子的走法

        # 送子逻辑：AI 把高价值子走到红方棋子攻击范围内
        """
        red_attacks = self._get_red_attack_squares(game)
        for fr, fc, tr, tc in moves:
            piece = game.board[fr][fc]
            if piece in _HIGH_VALUE_PIECES and (tr, tc) in red_attacks:
                return (fr, fc, tr, tc)
        return None

    def _get_red_attack_squares(self, game: ChessGame) -> set:
        """获取红方下一步能攻击到的所有格子"""
        squares = set()
        saved_turn = game.current_turn
        game.current_turn = RED
        for r in range(10):
            for c in range(9):
                if game.board[r][c] > 0:  # 红方棋子
                    for tr, tc in game.get_legal_moves(r, c):
                        squares.add((tr, tc))
        game.current_turn = saved_turn
        return squares

    def _score_all_moves(
        self,
        game: ChessGame,
        moves: List[Tuple],
    ) -> List[Tuple[int, Tuple]]:
        """对所有走法做浅层评分（深度1），返回按 score 升序列表"""
        scored = []
        for move in moves:
            g = game.board_copy()
            g.make_move(*move)
            score = self._evaluate(g)
            scored.append((score, move))
        scored.sort(key=lambda x: x[0])  # 升序：越小对 BLACK 越好
        return scored

    # ── 正常搜索 ──────────────────────────────────────────────────────────────

    def _best_move(
        self,
        game: ChessGame,
        moves: List[Tuple],
        depth: int,
        skill_score: float,
    ) -> Tuple[int, int, int, int]:
        """Alpha-Beta 搜索，搜索完成后根据棋力分加入轻微次优随机"""
        self._nodes = 0
        alpha, beta = -999999, 999999
        moves.sort(key=lambda m: abs(game.board[m[2]][m[3]]), reverse=True)

        scored = []
        for move in moves:
            g = game.board_copy()
            g.make_move(*move)
            score = self._minimax(g, depth - 1, alpha, beta, True)
            scored.append((score, move))
            beta = min(beta, score)

        if not scored:
            return random.choice(moves)

        scored.sort(key=lambda x: x[0])  # 越小对 BLACK 越好

        # 棋力分 40~80 时：有小概率选次优（保留一定人性化失误）
        if skill_score < 80 and len(scored) > 1:
            subopt_prob = max(0.0, (80 - skill_score) / 80 * 0.25)
            if random.random() < subopt_prob:
                top2 = scored[:min(3, len(scored))]
                return random.choice(top2)[1]

        return scored[0][1]

    def _minimax(self, game, depth, alpha, beta, is_max) -> int:
        self._nodes += 1
        if game.game_over or depth == 0:
            return self._evaluate(game)
        moves = game.get_all_moves(RED if is_max else BLACK)
        if not moves:
            return self._evaluate(game)
        moves.sort(key=lambda m: abs(game.board[m[2]][m[3]]), reverse=True)
        if is_max:
            best = -999999
            for move in moves:
                g = game.board_copy()
                g.make_move(*move)
                best = max(best, self._minimax(g, depth-1, alpha, beta, False))
                alpha = max(alpha, best)
                if beta <= alpha: break
            return best
        else:
            best = 999999
            for move in moves:
                g = game.board_copy()
                g.make_move(*move)
                best = min(best, self._minimax(g, depth-1, alpha, beta, True))
                beta = min(beta, best)
                if beta <= alpha: break
            return best

    def _evaluate(self, game: ChessGame) -> int:
        """局面评估（正值 RED 有利，负值 BLACK 有利）"""
        if game.game_over:
            return 100000 if game.winner == RED else -100000
        score = 0
        for r in range(10):
            for c in range(9):
                p = game.board[r][c]
                if p != 0:
                    score += PIECE_VALUES.get(p, 0)
                    score += _pst_score(p, r, c)
        return score
