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

放水走法选择（_blunder_move）：
  棋力分 < 15：优先尝试真正的“送子”（_find_giveaway_move，确保玩家下一步能直接吃到高价值子）
  其余情况：用局面评估器（NN 优先，无 NN 时用材料+位置分）给所有合法着法打分排序，
            再根据 skill_score 连续计算一个“目标着法百分位 + 随机窗口”，
            skill_score 越低 → 目标百分位越靠近“对黑方最差”的一端、窗口越宽；
            skill_score 越高 → 目标百分位越靠近最优着法、窗口越窄。
            这样放水力度是连续可调的，不再是离散分档，AI 的失误也更符合当前局面。
"""

from __future__ import annotations
import os
import random
import time
from typing import List, Optional, Tuple, Dict

import numpy as np

import config
from chess_game import (
    ChessGame, RED, BLACK,
    R_KING, R_GUARD, R_BISHOP, R_ROOK, R_KNIGHT, R_CANNON, R_PAWN,
    B_KING, B_GUARD, B_BISHOP, B_ROOK, B_KNIGHT, B_CANNON, B_PAWN,
    PIECE_VALUES, EMPTY,
)
from nn_evaluator import load_nn_evaluator, evaluate_board, NNEvaluator

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

# MicroDualNet 输出 v∈[-1,1]，与手工评估分（材料+位置分，量级约几千）混合前
# 需要放大到同一量级；一车（1000分）大致对应胜率评估中的显著优势，取该值作为缩放基准。
_MICRO_EVAL_SCALE = 1000.0

# ── 根节点震荡抑制（见 _best_move）──────────────────────────────────────────
# 会导致回到最近出现过局面（ChessGame.recent_keys）的招法，轻微降权而非禁用：
# 40 分远小于一兵（100分），不足以让 AI 放弃明显更优的招法，只在双方分数
# 本就接近、容易陷入来回震荡的局面里起作用。
_REPETITION_PENALTY = 40
# 两招分数差小于此阈值视为"打平"，随机二选一，避免确定性搜索永远选中同一招
# （同一开局/同一残局反复对弈时更容易触发来回震荡型假平局）。
_TIE_EPSILON = 10


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


def _score_to_ai_params(skill_score: float) -> Tuple[int, float, float, float]:
    """
    根据实时棋力分计算 AI 参数
    返回 (search_depth, blunder_rate, temperature, value_noise_std)

    # 棋力分越高 → 搜索深度越深，放水概率越低
    # 目标胜率 70%~90%：blunder_rate 为主要调节旋钮
    # temperature / value_noise_std：仅在 MicroDualNet 可用时生效（放水增强），
    #   skill_score 越低 → 温度越高（采样越随机）、噪声越大
    """
    frac = max(0.0, min(1.0, skill_score / 100.0))
    temperature = 0.05 + 0.35 * (1.0 - frac)      # skill 100 → 0.05，skill 0 → 0.40
    value_noise_std = 300.0 * (1.0 - frac)        # skill 100 → 0，skill 0 → 300

    if skill_score < 20:
        return 1, 0.70, temperature, value_noise_std   # 棋力分 0~20：深度1，放水70%
    if skill_score < 40:
        return 1, 0.55, temperature, value_noise_std   # 棋力分 20~40：深度1，放水55%
    if skill_score < 60:
        return 2, 0.35, temperature, value_noise_std   # 棋力分 40~60：深度2，放水35%
    if skill_score < 80:
        return 2, 0.18, temperature, value_noise_std   # 棋力分 60~80：深度2，放水18%
    return 3, 0.05, temperature, value_noise_std        # 棋力分 80~100：深度3，放水5%


class AIEngine:
    """
    自适应放水 AI 引擎

    每次 get_move() 接收实时 skill_score，动态决定：
      1. 是否触发放水（概率 = blunder_rate）
      2. 放水级别（RANDOM / GIVEAWAY / SUBOPTIMAL / MINOR）
      3. 正常搜索深度（search_depth）

    将 ai_blunder_rate / ai_search_depth 回写到 stats，供 HUD 显示
    """

    def __init__(self, model_path: Optional[str] = None, disable_nn: bool = False):
        """
        disable_nn=True：跳过所有神经网络加载（CNN + MicroDualNet），强制纯手工评估。
        供 gen_data.py 自我对弈数据生成时使用（多进程 CPU worker，不需要也不应加载网络）。
        """
        self._nodes = 0
        # 置换表：{board_key: (depth, score, best_move)}
        self._tt: Dict[Tuple, Tuple[int, int, Optional[Tuple[int,int,int,int]]]] = {}
        # 历史启发表：{move_key: score}
        self._history: Dict[Tuple[int,int,int,int], int] = {}
        self._nn_model = None
        self._nn_device = None
        self._nn_y_mean = 0.0
        self._nn_y_std = 1.0

        # ── MicroDualNet 占位（默认不可用，disable_nn 或加载失败时保持此状态）───
        self.nn_evaluator = NNEvaluator.__new__(NNEvaluator)
        self.nn_evaluator.nn_available = False
        self.nn_evaluator.model = None
        self.nn_evaluator.device = None
        self.nn_available = False

        # ── 放水增强参数（由 _score_to_ai_params 按技能分动态计算并回写）────────
        self.temperature = 0.1
        self.value_noise_std = 0.0

        if disable_nn:
            return

        if getattr(config, 'USE_NN_EVALUATOR', False):
            cnn_model_path = getattr(config, 'NN_EVAL_MODEL_PATH', 'model.pth')
            if os.path.exists(cnn_model_path):
                try:
                    self._nn_model, self._nn_device, self._nn_y_mean, self._nn_y_std = load_nn_evaluator(cnn_model_path)
                    print(f"Loaded NN evaluator: {cnn_model_path} on {self._nn_device}")
                except Exception as e:
                    print(f"Warning: failed to load NN evaluator {cnn_model_path}: {e}")
            else:
                print(f"Warning: NN evaluator model path not found: {cnn_model_path}")

        # ── MicroDualNet（轻量策略/价值双头网络，自我对弈训练产出）──────────────
        # 优先级最高的评估/排序来源；不存在 micro_dual.pth 时 nn_available=False，
        # 所有相关逻辑自动回退到旧的 CNN / manual 评估，行为与改造前完全一致。
        micro_model_path = model_path or getattr(config, 'MICRO_NN_MODEL_PATH', 'micro_dual.pth')
        try:
            self.nn_evaluator = NNEvaluator(micro_model_path)
            self.nn_available = self.nn_evaluator.nn_available
        except Exception as e:
            print(f"Warning: failed to init MicroDualNet evaluator {micro_model_path}: {e}")
        if self.nn_available:
            print(f"Loaded MicroDualNet evaluator: {micro_model_path}")

    def get_move(
        self,
        game: ChessGame,
        skill_score: float,
        stats=None,          # RehabStats，用于回写 AI 参数供 HUD 显示
        color: int = BLACK,  # 添加颜色参数，默认黑方保持向后兼容
    ) -> Optional[Tuple[int, int, int, int]]:
        
        """
        返回 AI 选择的走法 (fr, fc, tr, tc)

        # 实时棋力分: skill_score
        # AI放水率:   blunder_rate（见 _score_to_ai_params）
        # 强化学习奖励值: 由 AssessmentModule 在 main.py 中记录
        """
        moves = game.get_all_moves(color)
        if not moves:
            return None

        depth, blunder_rate, temperature, value_noise_std = _score_to_ai_params(skill_score)
        self.temperature = temperature
        self.value_noise_std = value_noise_std

        # 回写 AI 参数到 stats，HUD 可展示
        if stats is not None:
            stats.ai_blunder_rate = blunder_rate   # 注释：当前AI放水率
            stats.ai_search_depth = depth          # 注释：当前AI搜索深度

        # ── 放水决策 ──────────────────────────────────────────────────────────
        if random.random() < blunder_rate:
            return self._blunder_move(game, moves, skill_score)

        # ── 正常 Alpha-Beta 搜索 ──────────────────────────────────────────────
        return self._best_move(game, moves, depth, skill_score, color)

    # ── 放水走法生成 ──────────────────────────────────────────────────────────

    def _blunder_move(
        self,
        game: ChessGame,
        moves: List[Tuple],
        skill_score: float,
    ) -> Tuple[int, int, int, int]:
        """
        连续放水：用局面评估器给所有着法打分排序，再按 skill_score 在
        “百分位 + 随机窗口”上连续插值，选出一个故意变差的着法。

        排序后 scored[0] 是对 BLACK（AI）最好的着法，scored[-1] 是对 BLACK
        最差（对玩家最有利）的着法。

        # target_pct：目标百分位。skill_score 越低，越靠近 1.0（最差）；
        #             skill_score 越高，越靠近 0.0（最好）。
        # window：随机窗口半宽。skill_score 越低窗口越宽（失误更随机、更明显）；
        #         skill_score 越高窗口越窄（只在最优附近做很小的失误）。
        """
        if skill_score < 15:
            # 棋力分极低：优先尝试真正的送子，让玩家能直接吃到高价值子
            giveaway = self._find_giveaway_move(game, moves)
            if giveaway:
                return giveaway

        scored = self._score_all_moves(game, moves)
        n = len(scored)
        if n <= 1:
            return scored[0][1] if scored else random.choice(moves)

        target_pct = max(0.0, min(1.0, 1.0 - skill_score / 100.0))
        window = 0.5 - 0.42 * (skill_score / 100.0)  # 高分~0.08 ~ 低分~0.5

        lo_pct = max(0.0, target_pct - window)
        hi_pct = min(1.0, target_pct + window)
        lo_idx = int(round(lo_pct * (n - 1)))
        hi_idx = int(round(hi_pct * (n - 1)))
        if hi_idx <= lo_idx:
            hi_idx = min(n - 1, lo_idx + 1)

        candidates = scored[lo_idx:hi_idx + 1]
        return random.choice(candidates)[1]

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
    # ── 辅助：置换表 / 走法排序 / 静态搜索 ─────────────────────────────────────────

    def _board_key(self, game: ChessGame) -> Tuple:
        return (tuple(tuple(row) for row in game.board), game.current_turn)

    def _order_moves(self, game: ChessGame, moves: List[Tuple], use_nn_sorting: bool = False) -> List[Tuple]:
        """
        走法排序。
          use_nn_sorting=True 且 MicroDualNet 可用时（仅根节点调用）：
            用策略网络对着法打分排序，网络失败时自动回退到手工排序。
          其余情况（内部节点，或网络不可用）：吃子优先 + 历史启发排序（降序），
          与改造前完全一致。
        """
        if use_nn_sorting and self.nn_available:
            try:
                move_scores = self.nn_evaluator.policy_scores(game, moves)
                sign = 1.0 if game.current_turn == RED else -1.0
                ordered = sorted(moves, key=lambda m: sign * move_scores.get(m, 0.0), reverse=True)
                return ordered
            except Exception:
                pass

        scored = []
        for m in moves:
            fr, fc, tr, tc = m
            captured = game.board[tr][tc]
            cap_score = abs(captured) if captured != EMPTY else 0
            hist_score = self._history.get(m, 0)
            score = cap_score * 1000 + hist_score
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _quiescence(self, game: ChessGame, alpha: int, beta: int, color: int) -> int:
        """简单静态搜索：只扩展吃子走法以稳定评估"""
        stand_pat = self._evaluate(game)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        moves = game.get_all_moves(color)
        # 只保留吃子走法
        cap_moves = [m for m in moves if game.board[m[2]][m[3]] != EMPTY]
        if not cap_moves:
            return stand_pat

        cap_moves = self._order_moves(game, cap_moves)
        for move in cap_moves:
            g = game.board_copy()
            g.make_move(*move)
            score = -self._quiescence(g, -beta, -alpha, -color)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # ── 正常搜索（迭代加深 + 置换表 + Alpha-Beta）────────────────────────────────

    def _root_ordered_moves(self, game: ChessGame, moves: List[Tuple], skill_score: float) -> List[Tuple]:
        """
        根节点走法排序 + 候选裁剪。
        MicroDualNet 可用时：用策略分数排序，并只保留 Top-K（K 随技能分调整）进入搜索；
        否则完全回退到原有的手工排序，不做裁剪（行为与改造前一致）。
        """
        ordered = self._order_moves(game, moves)
        if not self.nn_available:
            return ordered
        try:
            policy_scores = self.nn_evaluator.policy_scores(game, moves)
            ordered = sorted(moves, key=lambda m: policy_scores.get(m, 0.0), reverse=True)
            k = max(6, min(len(ordered), 10 + int(skill_score / 10)))  # K 随技能分调整，默认约 15
            return ordered[:k]
        except Exception:
            return ordered

    def _best_move(
        self,
        game: ChessGame,
        moves: List[Tuple],
        depth: int,
        skill_score: float,
        color: int = BLACK,
    ) -> Tuple[int, int, int, int]:
        """
        迭代加深调用 _minimax，并使用历史启发/置换表提高效率。

        按 color 感知根节点的 max/min 方向（与 get_move_with_scores 一致）：
        RED 求最大、BLACK 求最小，_minimax 的返回值始终是“正值对 RED 有利”。
        历史遗留写法曾在此处硬编码 is_max=False 且始终取 `score > best`，
        只在根为 RED 时正确；根为 BLACK（AI 默认执黑）时会系统性地
        选出对红方更有利、而非对黑方更有利的招法，现已按 is_red 修正。
        """
        self._nodes = 0
        is_red = (color == RED)
        best_move = None
        best_score = None

        # 初始走法排序（网络可用时使用策略网络排序 + Top-K 裁剪，否则用原手工排序）
        ordered_moves = self._root_ordered_moves(game, moves, skill_score)
        root_scores: Dict[Tuple[int, int, int, int], int] = {}
        # 走这步前最近出现过的局面（含当前局面），见 _REPETITION_PENALTY
        recent_keys = set(game.recent_keys)

        for d in range(1, max(1, depth) + 1):
            alpha, beta = -999999, 999999
            local_best = None
            local_best_score = None
            for move in ordered_moves:
                g = game.board_copy()
                g.make_move(*move)
                # 走完这步后轮到对方，故 is_max 对应对方是否为 RED
                score = self._minimax(g, d - 1, alpha, beta, not is_red)
                # root_scores 记录原始评估分（正值 RED 有利），供下方温度采样使用
                root_scores[move] = score

                # 走完这步会回到最近出现过的局面时轻微降权：抑制无历史感知的
                # 确定性搜索反复选中同一组"来回走"招法、陷入死循环导致假平局。
                adj_score = score
                if g.board_key() in recent_keys:
                    adj_score = score - _REPETITION_PENALTY if is_red else score + _REPETITION_PENALTY

                if local_best_score is None:
                    local_best_score = adj_score
                    local_best = move
                else:
                    better = (adj_score > local_best_score) if is_red else (adj_score < local_best_score)
                    # 分数打平时随机二选一，避免确定性搜索永远选中排序中更靠前的同一招
                    tie = abs(adj_score - local_best_score) <= _TIE_EPSILON
                    if better or (tie and random.random() < 0.5):
                        local_best_score = adj_score
                        local_best = move

                if is_red:
                    alpha = max(alpha, score)
                else:
                    beta = min(beta, score)
            if local_best is not None:
                best_move = local_best
                best_score = local_best_score

        if not best_move:
            return random.choice(moves)

        # 棋力分 40~80 时：有小概率选次优（保留一定人性化失误）
        if skill_score < 80 and len(ordered_moves) > 1:
            subopt_prob = max(0.0, (80 - skill_score) / 80 * 0.25)
            if random.random() < subopt_prob:
                top_choices = ordered_moves[:min(3, len(ordered_moves))]
                return random.choice(top_choices)

        # ── 放水增强：网络可用时，对根节点走法按温度采样而非严格取最大分 ──────
        if self.nn_available and self.temperature > 0 and len(root_scores) > 1:
            try:
                return self._temperature_sample(root_scores, best_move, is_red)
            except Exception:
                pass

        return best_move

    def _temperature_sample(
        self,
        root_scores: Dict[Tuple[int, int, int, int], int],
        fallback_move: Tuple[int, int, int, int],
        is_red: bool,
    ) -> Tuple[int, int, int, int]:
        """
        对根节点走法按温度做 softmax 采样。
        root_scores 为原始评估分（正值 RED 有利）：RED 分数越大越优先，
        BLACK 分数越小越优先，采样前先按 is_red 转成“越大越优先”的偏好分。
        """
        items = list(root_scores.items())
        moves_arr = [m for m, _ in items]
        raw = np.array([s for _, s in items], dtype=np.float64)
        prefs = raw if is_red else -raw
        prefs = prefs - prefs.max()
        scale = max(self.temperature, 1e-3) * 1000.0  # 分数量级约几百~几千，需要缩放温度
        weights = np.exp(prefs / scale)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return fallback_move
        probs = weights / total
        idx = np.random.choice(len(moves_arr), p=probs)
        return moves_arr[idx]

    def _minimax(self, game: ChessGame, depth: int, alpha: int, beta: int, is_max: bool) -> int:
        self._nodes += 1
        key = self._board_key(game)
        # 置换表查询
        tt_entry = self._tt.get(key)
        if tt_entry is not None:
            stored_depth, stored_score, _ = tt_entry
            if stored_depth >= depth:
                return stored_score

        if game.game_over:
            val = 100000 if game.winner == RED else -100000
            return val

        if depth <= 0:
            # 进入静态搜索以避免“快照问题”
            val = self._quiescence(game, alpha, beta, RED if is_max else BLACK)
            # 记录到置换表
            self._tt[key] = (0, val, None)
            return val

        moves = game.get_all_moves(RED if is_max else BLACK)
        if not moves:
            val = self._evaluate(game)
            self._tt[key] = (depth, val, None)
            return val

        moves = self._order_moves(game, moves)

        if is_max:
            best = -999999
            best_move = None
            for move in moves:
                g = game.board_copy()
                g.make_move(*move)
                score = self._minimax(g, depth - 1, alpha, beta, False)
                if score > best:
                    best = score
                    best_move = move
                alpha = max(alpha, best)
                if beta <= alpha:
                    # 历史启发：增加该走法权重
                    self._history[move] = self._history.get(move, 0) + (1 << depth)
                    break
            self._tt[key] = (depth, best, best_move)
            return best
        else:
            best = 999999
            best_move = None
            for move in moves:
                g = game.board_copy()
                g.make_move(*move)
                score = self._minimax(g, depth - 1, alpha, beta, True)
                if score < best:
                    best = score
                    best_move = move
                beta = min(beta, best)
                if beta <= alpha:
                    self._history[move] = self._history.get(move, 0) + (1 << depth)
                    break
            self._tt[key] = (depth, best, best_move)
            return best

    # ── 自我对弈数据生成专用接口 ──────────────────────────────────────────────

    def get_move_with_scores(
        self,
        game: ChessGame,
        depth: int = 2,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[Tuple[int, int, int, int], int]]:
        """
        供 gen_data.py 自我对弈使用：返回 (最佳走法, {所有合法走法: 原始搜索评分})。

        与 _best_move 独立实现（不复用其根节点 is_max=False 的历史遗留写法），
        对当前走棋方（game.current_turn，可为 RED 或 BLACK）分别正确处理：
          - RED 走子：评分越大越好；
          - BLACK 走子：评分越小越好（因为 _minimax 的返回值始终是“正值对 RED 有利”）。
        评分为 _evaluate 的原始输出（正值 RED 有利），与 _minimax/_quiescence 使用同一套评估，
        不受放水温度/噪声影响（自我对弈数据需要干净的搜索评分作为训练目标）。
        """
        color = game.current_turn
        moves = game.get_all_moves(color)
        if not moves:
            return None, {}

        is_red = (color == RED)
        ordered_moves = self._order_moves(game, moves)
        scores: Dict[Tuple[int, int, int, int], int] = {}
        best_move = None
        best_score = None

        alpha, beta = -999999, 999999
        for move in ordered_moves:
            g = game.board_copy()
            g.make_move(*move)
            # 走完这步后轮到对方，故 is_max 对应对方是否为 RED
            score = self._minimax(g, max(0, depth - 1), alpha, beta, not is_red)
            scores[move] = score
            if best_score is None or (score > best_score if is_red else score < best_score):
                best_score = score
                best_move = move
            if is_red:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)

        return best_move, scores

    def _manual_evaluate(self, game: ChessGame) -> int:
        """纯手工局面评估（材料 + 位置分），正值 RED 有利。不依赖任何网络，始终可用。"""
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

    def _evaluate(self, game: ChessGame) -> int:
        """
        局面评估（正值 RED 有利，负值 BLACK 有利）。优先级：
          1. MicroDualNet（若可用）：0.7 * nn_v(缩放) + 0.3 * manual_v 混合评估
          2. 旧 CNN 评估器（若可用）：直接返回其输出（与改造前行为一致）
          3. 纯手工评估 _manual_evaluate（始终可用的回退）
        棋局终局始终直接返回 ±100000，不受网络影响。
        """
        if game.game_over:
            return 100000 if game.winner == RED else -100000

        if self.nn_available:
            try:
                nn_v = self.nn_evaluator.value(game)          # in [-1, 1]，正值对 RED 有利
                manual_v = self._manual_evaluate(game)
                # MicroDualNet 的 tanh 输出与手工评估的分数量级不同，需缩放到可比的量级
                # 再按 0.7/0.3 权重混合，否则网络的贡献会被手工分数的数值量级淹没。
                value = 0.7 * (nn_v * _MICRO_EVAL_SCALE) + 0.3 * manual_v
                if self.value_noise_std > 0:
                    value += float(np.random.normal(0, self.value_noise_std))
                return int(value)
            except Exception:
                pass

        if self._nn_model is not None and self._nn_device is not None:
            try:
                value = evaluate_board(game, self._nn_model, self._nn_device, self._nn_y_mean, self._nn_y_std)
                if self.value_noise_std > 0:
                    value += float(np.random.normal(0, self.value_noise_std))
                return int(value)
            except Exception:
                pass

        value = self._manual_evaluate(game)
        if self.value_noise_std > 0:
            value += float(np.random.normal(0, self.value_noise_std))
        return int(value)
