# -*- coding: utf-8 -*-
"""
康复评估模块 —— Q-Learning 实时棋力评分
棋力分 skill_score: 0~100，实时量化用户实力
Q-Learning 状态特征：(失误率档位, 吃子效率档位, 决策速度档位, 走棋合法性档位)
Q 值含义：当前状态下对棋力分的期望调整量（正=升分，负=降分）
"""

from __future__ import annotations
import time
import random
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import config


# ── Q-Learning 超参数 ─────────────────────────────────────────────────────────
QL_ALPHA       = 0.15   # 学习率：每步对 Q 表的更新步长
QL_GAMMA       = 0.80   # 折扣因子：未来奖励的权重
QL_EPSILON     = 0.10   # ε-贪婪探索率（10% 随机探索防过拟合）

# 奖励设计（正奖励=升分，负奖励=降分）
R_LEGAL_MOVE   = +2.0   # 合法走棋
R_CAPTURE      = +5.0   # 吃子（展现棋力）
R_FAST_MOVE    = +3.0   # 快速决策（手部控制灵活）
R_SLOW_MOVE    = -1.0   # 慢速决策
R_ILLEGAL      = -3.0   # 非法走棋/手势失误
R_WIN_BONUS    = +15.0  # 用户赢棋奖励
R_LOSE_PENALTY = -8.0   # 用户输棋惩罚（但幅度小于赢棋，鼓励参与）

# 决策速度分界（秒）
FAST_THRESHOLD = 8.0    # 快速：< 8s
SLOW_THRESHOLD = 20.0   # 慢速：> 20s

# 棋力分边界
SCORE_MIN = 0.0
SCORE_MAX = 100.0
SCORE_INIT = 30.0       # 初始分（偏低，给用户上升空间）


@dataclass
class RehabStats:
    # ── 基础统计 ──────────────────────────────────────────────────────────────
    total_moves:      int   = 0
    illegal_attempts: int   = 0
    captures:         int   = 0
    session_seconds:  float = 0.0
    avg_move_time:    float = 0.0
    error_rate:       float = 0.0

    # ── Q-Learning 棋力评分 ───────────────────────────────────────────────────
    skill_score:      float = SCORE_INIT   # 实时棋力分 0~100
    skill_rating:     int   = 1            # 星级 1-5（兼容旧接口）
    skill_label:      str   = "初学康复"

    # ── AI 放水参数（由 AIEngine 回写，用于 HUD 显示）────────────────────────
    ai_blunder_rate:  float = 0.50         # 当前 AI 放水概率
    ai_search_depth:  int   = 1            # 当前 AI 搜索深度
    last_ql_reward:   float = 0.0          # 上一步 Q-Learning 奖励值

    # ── 手势统计 ──────────────────────────────────────────────────────────────
    gesture_attempts: int   = 0
    gesture_success:  int   = 0

    @property
    def gesture_accuracy(self) -> float:
        if self.gesture_attempts == 0:
            return 1.0
        return self.gesture_success / self.gesture_attempts


SKILL_LABELS = {
    1: "初学康复",   # 0~20
    2: "进阶练习",   # 20~40
    3: "稳定发展",   # 40~60
    4: "熟练掌握",   # 60~80
    5: "康复达人",   # 80~100
}


def _score_to_rating(score: float) -> int:
    """棋力分 → 星级（1-5）"""
    if score < 20: return 1
    if score < 40: return 2
    if score < 60: return 3
    if score < 80: return 4
    return 5


# ── 状态离散化（Q 表键） ──────────────────────────────────────────────────────

def _error_bucket(error_rate: float) -> int:
    """失误率 → 0/1/2（低/中/高）"""
    if error_rate < 0.10: return 0
    if error_rate < 0.25: return 1
    return 2

def _capture_bucket(captures: int, moves: int) -> int:
    """吃子效率 = captures/moves → 0/1/2（低/中/高）"""
    rate = captures / max(moves, 1)
    if rate < 0.05: return 0
    if rate < 0.15: return 1
    return 2

def _speed_bucket(avg_time: float) -> int:
    """决策速度 → 0/1/2（快/中/慢）"""
    if avg_time < FAST_THRESHOLD:  return 0
    if avg_time < SLOW_THRESHOLD:  return 1
    return 2

def _legality_bucket(error_rate: float) -> int:
    """走棋合法性（与失误率高度相关，但单独分桶强调该维度）"""
    if error_rate < 0.05: return 0
    if error_rate < 0.20: return 1
    return 2

def _make_state(s: RehabStats) -> Tuple[int, int, int, int]:
    """
    Q 表状态键：(失误率桶, 吃子效率桶, 速度桶, 合法性桶)
    共 3^4 = 81 个状态，轻量 Q 表
    """
    return (
        _error_bucket(s.error_rate),
        _capture_bucket(s.captures, s.total_moves),
        _speed_bucket(s.avg_move_time),
        _legality_bucket(s.error_rate),
    )

# 动作空间：棋力分调整幅度（离散化动作）
# 动作 0=-5, 1=-2, 2=0, 3=+2, 4=+5
_ACTIONS = [-5.0, -2.0, 0.0, +2.0, +5.0]


class AssessmentModule:
    """
    玩家表现评估器 + Q-Learning 棋力评分引擎

    Q 表：state → [Q(a0), Q(a1), Q(a2), Q(a3), Q(a4)]
    每步走棋后：
      1. 根据走棋特征计算即时奖励 r
      2. ε-贪婪选动作（棋力分调整量）
      3. 执行动作更新棋力分
      4. 用 TD 误差更新 Q 表

    注：Q 表在本局内实时学习，不跨局持久化（保证轻量）
    """

    def __init__(self):
        self._stats = RehabStats()
        self._session_start = time.time()
        self._move_times: List[float] = []
        self._last_move_start: Optional[float] = None

        # Q 表：Dict[state_tuple, List[float]] 共5个动作的 Q 值
        self._q_table: Dict[Tuple, List[float]] = {}

        # 上一步状态（用于 TD 更新）
        self._prev_state: Optional[Tuple] = None
        self._prev_action: Optional[int] = None

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def start_move_timer(self):
        self._last_move_start = time.time()

    def record_move(self, captured: bool = False):
        """
        记录一次成功走棋，触发 Q-Learning 更新
        # 即时奖励 = R_LEGAL_MOVE [+吃子奖励] [+速度奖励]
        """
        elapsed = 0.0
        if self._last_move_start is not None:
            elapsed = time.time() - self._last_move_start
            self._last_move_start = None
        self._move_times.append(elapsed)

        s = self._stats
        s.total_moves += 1
        s.gesture_success += 1
        s.gesture_attempts += 1
        if captured:
            s.captures += 1
        self._refresh_basics()

        # 计算即时奖励
        reward = R_LEGAL_MOVE
        if captured:
            reward += R_CAPTURE                         # 吃子奖励
        if elapsed < FAST_THRESHOLD:
            reward += R_FAST_MOVE                       # 快速决策奖励
        elif elapsed > SLOW_THRESHOLD:
            reward += R_SLOW_MOVE                       # 慢速惩罚

        # Q-Learning 更新
        # 注释：reward={reward:.1f}  skill_score={s.skill_score:.1f}
        self._ql_step(reward)

    def record_illegal_attempt(self):
        """
        记录非法操作，触发负奖励 Q 更新
        # 即时奖励 = R_ILLEGAL（降低棋力分）
        """
        s = self._stats
        s.illegal_attempts += 1
        s.gesture_attempts += 1
        self._refresh_basics()
        self._ql_step(R_ILLEGAL)

    def record_gesture_attempt(self):
        self._stats.gesture_attempts += 1

    def record_game_result(self, player_won: bool):
        """
        局末奖励：玩家赢棋给大额正奖励，输棋小额负奖励
        # 赢棋奖励 = +15.0，输棋惩罚 = -8.0
        """
        reward = R_WIN_BONUS if player_won else R_LOSE_PENALTY
        self._ql_step(reward)

    def get_stats(self) -> RehabStats:
        self._stats.session_seconds = time.time() - self._session_start
        return self._stats

    def reset(self):
        self._stats = RehabStats()
        self._session_start = time.time()
        self._move_times = []
        self._last_move_start = None
        self._q_table = {}
        self._prev_state = None
        self._prev_action = None

    # ── Q-Learning 核心 ───────────────────────────────────────────────────────

    def _ql_step(self, reward: float):
        """
        Q-Learning 单步更新
        TD 误差：δ = r + γ * max_a'Q(s',a') - Q(s,a)
        Q(s,a) ← Q(s,a) + α * δ

        # 当前棋力分: {self._stats.skill_score:.1f}
        # 即时奖励:   {reward:.2f}
        # 放水概率:   由 AIEngine 根据 skill_score 动态计算
        """
        s_curr = _make_state(self._stats)

        # ε-贪婪选动作
        if random.random() < QL_EPSILON:
            action_idx = random.randrange(len(_ACTIONS))   # 探索
        else:
            q_curr = self._get_q(s_curr)
            action_idx = q_curr.index(max(q_curr))          # 利用

        # 执行动作：更新棋力分
        delta_score = _ACTIONS[action_idx]
        old_score = self._stats.skill_score
        new_score = max(SCORE_MIN, min(SCORE_MAX, old_score + delta_score))
        # 奖励驱动：正奖励时棋力分上升，负奖励时下降（叠加奖励符号方向）
        reward_driven = reward * 0.4   # 奖励直接推动分数微调
        new_score = max(SCORE_MIN, min(SCORE_MAX, new_score + reward_driven))
        self._stats.skill_score = new_score
        self._stats.last_ql_reward = reward  # 注释：HUD 展示上一步奖励值

        # TD 更新前一步 Q 值
        if self._prev_state is not None and self._prev_action is not None:
            q_prev = self._get_q(self._prev_state)
            q_next_max = max(self._get_q(s_curr))
            td_error = reward + QL_GAMMA * q_next_max - q_prev[self._prev_action]
            q_prev[self._prev_action] += QL_ALPHA * td_error
            self._q_table[self._prev_state] = q_prev

        self._prev_state = s_curr
        self._prev_action = action_idx

        # 同步星级和标签
        self._stats.skill_rating = _score_to_rating(self._stats.skill_score)
        self._stats.skill_label  = SKILL_LABELS[self._stats.skill_rating]

    def _get_q(self, state: Tuple) -> List[float]:
        """获取 Q 表条目，不存在则初始化为全零"""
        if state not in self._q_table:
            self._q_table[state] = [0.0] * len(_ACTIONS)
        return self._q_table[state]

    # ── 内部基础统计刷新 ──────────────────────────────────────────────────────

    def _refresh_basics(self):
        s = self._stats
        if self._move_times:
            s.avg_move_time = sum(self._move_times) / len(self._move_times)
        total = s.total_moves + s.illegal_attempts
        s.error_rate = s.illegal_attempts / total if total else 0.0
