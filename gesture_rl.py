# -*- coding: utf-8 -*-
"""
手势强化学习模块 — GestureRL
核心目标：实时根据玩家手部动作质量自动调整 MediaPipe 捏合阈值，
          使阈值随康复进展动态收紧，持续施加适度训练压力。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
强化学习设计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【观测的两类失误】
  grab_miss  : 握紧手但起点无效（没有己方棋子）
               → 说明捏合阈值偏高，手稍微用力就误触发
  release_miss: 松开手但落点无效（棋盘外 / 无选中棋子）
               → 说明松开阈值偏低，手还没充分展开就触发

【状态空间】3×3×3 = 27 个离散状态
  s = (grab_miss_rate_bucket,   # 握紧失误率 0/1/2
       release_miss_rate_bucket,# 松开失误率 0/1/2
       recent_trend_bucket)     # 近期失误趋势 改善/持平/恶化 0/1/2

【动作空间】9 个动作，分别调整三个阈值
  0: pinch    -δ（更难捏合，训练握力）
  1: pinch    不变
  2: pinch    +δ（更易捏合，降低难度）
  3: release  -δ（更难松开）
  4: release  不变
  5: release  +δ（更易松开）
  6: hold     -1（减少需保持帧数，更灵敏）
  7: hold     不变
  8: hold     +1（增加需保持帧数，更稳定）

【奖励函数】
  有效抓取（GRABBING 且有棋子）    : +2.0  手部控制精准
  有效落子（RELEASING 且落点合法） : +3.0  完成完整动作
  握紧失误（grab_miss）           : -2.0  捏合误触，阈值需调整
  松开失误（release_miss）        : -1.5  松开误触，阈值需调整
  连续成功 N 步后额外奖励          : +1.0  鼓励稳定表现，此时收紧阈值

【阈值调整策略】
  失误率高 → 阈值宽松（降低训练难度，确保可操作性）
  失误率低 → 阈值收紧（增加训练难度，推进康复进展）
  连续成功  → 主动收紧阈值（核心康复机制）

【阈值安全范围】
  PINCH_THRESHOLD  : [0.08, 0.35]  防止过极端
  RELEASE_THRESHOLD: [0.15, 0.50]  始终 > PINCH + 0.05
  MIN_HOLD_FRAMES  : [1, 6]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ── 超参数 ────────────────────────────────────────────────────────────────────
RL_ALPHA   = 0.15    # Q-Learning 学习率
RL_GAMMA   = 0.85    # 折扣因子
RL_EPSILON = 0.12    # ε-贪婪探索率

# 阈值调整步长
DELTA_PINCH   = 0.02   # 捏合阈值每步变化量
DELTA_RELEASE = 0.02   # 松开阈值每步变化量

# 阈值安全边界
PINCH_MIN,   PINCH_MAX   = 0.08, 0.35
RELEASE_MIN, RELEASE_MAX = 0.15, 0.50
HOLD_MIN,    HOLD_MAX    = 1,    6

# 失误率分桶边界
MISS_LOW  = 0.10   # 失误率 < 10% → 低档
MISS_HIGH = 0.30   # 失误率 > 30% → 高档

# 近期观察窗口（事件数）
WINDOW_SIZE = 20

# 连续成功多少次后主动收紧阈值
STREAK_THRESHOLD = 8

# 即时奖励
R_GRAB_SUCCESS    = +2.0
R_RELEASE_SUCCESS = +3.0
R_GRAB_MISS       = -2.0
R_RELEASE_MISS    = -1.5
R_STREAK_BONUS    = +1.0


@dataclass
class GestureRLStats:
    """供 HUD 展示的 RL 状态快照"""
    pinch_threshold:    float = 0.20
    release_threshold:  float = 0.30
    hold_frames:        int   = 2
    grab_miss_rate:     float = 0.0   # 握紧失误率（近期窗口）
    release_miss_rate:  float = 0.0   # 松开失误率（近期窗口）
    last_reward:        float = 0.0   # 上一步 RL 奖励
    success_streak:     int   = 0     # 当前连续成功次数
    total_adjustments:  int   = 0     # 累计阈值调整次数
    rl_action_name:     str   = "-"   # 上一次执行的动作描述


# 动作描述（用于 HUD 展示）
_ACTION_NAMES = [
    "捏合↑难", "捏合持平", "捏合↓易",
    "松开↑难", "松开持平", "松开↓易",
    "保持↑稳", "保持持平", "保持↓灵",
]

# 动作数量
N_ACTIONS = 9


def _bucket(rate: float) -> int:
    """失误率 → 0(低)/1(中)/2(高)"""
    if rate < MISS_LOW:  return 0
    if rate < MISS_HIGH: return 1
    return 2


def _trend_bucket(window: deque) -> int:
    """
    近期事件趋势 → 0(改善)/1(持平)/2(恶化)
    比较前半窗口和后半窗口的失误比例
    """
    if len(window) < 6:
        return 1
    half = len(window) // 2
    items = list(window)
    # 事件：True=成功，False=失误
    early_miss = sum(1 for x in items[:half] if not x) / half
    late_miss  = sum(1 for x in items[half:] if not x) / (len(items) - half)
    diff = late_miss - early_miss
    if diff < -0.05:  return 0   # 改善
    if diff >  0.05:  return 2   # 恶化
    return 1                      # 持平


class GestureRL:
    """
    手势强化学习控制器

    使用方式（main.py 每帧调用）：
      rl = GestureRL(init_pinch, init_release, init_hold)

      # 每次发生手势事件时上报：
      rl.record_grab_success()
      rl.record_grab_miss()
      rl.record_release_success()
      rl.record_release_miss()

      # 每帧将最新阈值注入 VisionModule：
      vision.set_thresholds(rl.pinch_threshold,
                            rl.release_threshold,
                            rl.hold_frames)

      # 获取 HUD 统计：
      stats = rl.get_stats()
    """

    def __init__(
        self,
        init_pinch:   float = 0.20,
        init_release: float = 0.30,
        init_hold:    int   = 2,
    ):
        # 当前阈值（实时注入 VisionModule）
        self.pinch_threshold:   float = init_pinch
        self.release_threshold: float = init_release
        self.hold_frames:       int   = init_hold

        # Q 表：Dict[state_tuple, List[float]]
        self._q: Dict[Tuple[int, int, int], List[float]] = {}

        # 近期事件窗口（True=成功, False=失误）
        self._grab_window:    deque = deque(maxlen=WINDOW_SIZE)
        self._release_window: deque = deque(maxlen=WINDOW_SIZE)
        # 合并窗口用于趋势判断
        self._all_window:     deque = deque(maxlen=WINDOW_SIZE)

        # 连续成功计数
        self._success_streak: int = 0

        # 上一步 Q-Learning 信息（用于 TD 更新）
        self._prev_state:  Optional[Tuple[int, int, int]] = None
        self._prev_action: Optional[int] = None
        # 挂起动作：在下一步开始时应用（用于使某些失误的调整延迟生效）
        self._pending_action: Optional[int] = None

        # 统计
        self._last_reward:      float = 0.0
        self._total_adjustments: int  = 0
        self._last_action_name: str   = "-"

    # ── 事件上报接口 ──────────────────────────────────────────────────────────

    def record_grab_success(self):
        """握紧手且起点有效（成功选到棋子）"""
        # 奖励：+2.0  手部控制精准
        self._grab_window.append(True)
        self._all_window.append(True)
        self._success_streak += 1
        reward = R_GRAB_SUCCESS
        if self._success_streak >= STREAK_THRESHOLD:
            # 连续成功足够多次：额外奖励，同时主动收紧阈值
            reward += R_STREAK_BONUS
        self._step(reward)

    def record_grab_miss(self):
        """握紧手但起点无效（该位置没有己方棋子）
        # 奖励 = R_GRAB_MISS = -2.0
        # 说明：捏合阈值可能偏高，手稍微用力即误触发
        # RL 将倾向于选择 action=2（捏合↓易）或 action=6（保持↑稳）
        """
        self._grab_window.append(False)
        self._all_window.append(False)
        self._success_streak = 0
        self._step(R_GRAB_MISS)

    def record_release_success(self):
        """松开手且落点合法（成功走棋）
        # 奖励 = R_RELEASE_SUCCESS = +3.0
        """
        self._release_window.append(True)
        self._all_window.append(True)
        self._success_streak += 1
        reward = R_RELEASE_SUCCESS
        if self._success_streak >= STREAK_THRESHOLD:
            reward += R_STREAK_BONUS
        self._step(reward)

    def record_release_miss(self):
        """松开手但落点无效（棋盘外 / 无选中子 / 非法落点）
        # 奖励 = R_RELEASE_MISS = -1.5
        # 说明：松开阈值可能偏低，手未充分展开就触发落子
        # RL 将倾向于选择 action=5（松开↓易）或提高 hold_frames
        """
        self._release_window.append(False)
        self._all_window.append(False)
        self._success_streak = 0
        self._step(R_RELEASE_MISS)

    def record_dwell_timeout(self):
        """手在某位置停留3秒未能完成抓取或落子
        # 立即降低当前阈值并给 RL 负奖励
        # 说明：用户手部停在棋盘上但无法触发动作，说明当前阈值对该用户偏难
        """
        self._all_window.append(False)
        self._success_streak = 0
        self._force_lower_thresholds()
        self._step(-2.5)

    def _force_lower_thresholds(self) -> None:
        """停留超时后立即让手势阈值更易触发。"""
        changed = False
        changed |= self._apply_action(2)  # 让捏合更容易
        changed |= self._apply_action(5)  # 让松开更容易
        changed |= self._apply_action(6)  # 让保持更灵敏
        if changed:
            self._total_adjustments += 1

    def get_stats(self) -> GestureRLStats:
        grab_miss_rate    = self._miss_rate(self._grab_window)
        release_miss_rate = self._miss_rate(self._release_window)
        return GestureRLStats(
            pinch_threshold   = self.pinch_threshold,
            release_threshold = self.release_threshold,
            hold_frames       = self.hold_frames,
            grab_miss_rate    = grab_miss_rate,
            release_miss_rate = release_miss_rate,
            last_reward       = self._last_reward,
            success_streak    = self._success_streak,
            total_adjustments = self._total_adjustments,
            rl_action_name    = self._last_action_name,
        )

    # ── Q-Learning 核心 ───────────────────────────────────────────────────────

    def _get_state(self) -> Tuple[int, int, int]:
        """
        构造当前状态三元组
        # state = (握紧失误率桶, 松开失误率桶, 近期趋势桶)
        """
        return (
            _bucket(self._miss_rate(self._grab_window)),
            _bucket(self._miss_rate(self._release_window)),
            _trend_bucket(self._all_window),
        )

    def _step(self, reward: float):
        """
        Q-Learning 单步：
          1. 观测当前状态 s'
          2. TD 更新上一步 Q(s,a)
          3. ε-贪婪选新动作 a'
          4. 执行动作（修改阈值）
        # RL奖励值: reward
        """
        # 先应用上一步挂起的动作（如果有），使某些调整延迟到当前步骤生效
        if self._pending_action is not None:
            changed = self._apply_action(self._pending_action)
            if changed:
                self._total_adjustments += 1
            # 清除挂起
            self._pending_action = None

        self._last_reward = reward
        s_next = self._get_state()

        # TD 更新
        if self._prev_state is not None and self._prev_action is not None:
            q_prev = self._get_q(self._prev_state)
            q_next_max = max(self._get_q(s_next))
            td = reward + RL_GAMMA * q_next_max - q_prev[self._prev_action]
            q_prev[self._prev_action] += RL_ALPHA * td
            self._q[self._prev_state] = q_prev

        # ε-贪婪选动作（但不立即执行，调度到下一步生效）
        if random.random() < RL_EPSILON:
            action = random.randrange(N_ACTIONS)
        else:
            q_curr = self._get_q(s_next)
            action = q_curr.index(max(q_curr))

        # 将动作挂起，在下一次 _step() 开头应用
        self._pending_action = action

        self._last_action_name = _ACTION_NAMES[action]
        self._prev_state  = s_next
        self._prev_action = action

    def _apply_action(self, action: int) -> bool:
        """
        执行阈值调整动作，返回是否实际发生变化

        动作编码：
          0/1/2 → 调整 pinch_threshold (-δ / 不变 / +δ)
          3/4/5 → 调整 release_threshold (-δ / 不变 / +δ)
          6/7/8 → 调整 hold_frames (-1 / 不变 / +1)

        安全约束：
          release_threshold 始终 > pinch_threshold + 0.05
        # 阈值调整：action={action}, pinch={self.pinch_threshold:.3f},
        #           release={self.release_threshold:.3f}, hold={self.hold_frames}
        """
        old_p = self.pinch_threshold
        old_r = self.release_threshold
        old_h = self.hold_frames

        if action == 0:
            self.pinch_threshold = max(PINCH_MIN, self.pinch_threshold - DELTA_PINCH)
        elif action == 2:
            self.pinch_threshold = min(PINCH_MAX, self.pinch_threshold + DELTA_PINCH)
        elif action == 3:
            self.release_threshold = max(RELEASE_MIN, self.release_threshold - DELTA_RELEASE)
        elif action == 5:
            self.release_threshold = min(RELEASE_MAX, self.release_threshold + DELTA_RELEASE)
        elif action == 6:
            self.hold_frames = max(HOLD_MIN, self.hold_frames - 1)
        elif action == 8:
            self.hold_frames = min(HOLD_MAX, self.hold_frames + 1)
        # action 1/4/7：不变

        # 安全间距约束：松开阈值必须比捏合阈值大至少 0.05
        if self.release_threshold < self.pinch_threshold + 0.05:
            self.release_threshold = round(self.pinch_threshold + 0.05, 3)
            self.release_threshold = min(RELEASE_MAX, self.release_threshold)

        return (self.pinch_threshold   != old_p or
                self.release_threshold != old_r or
                self.hold_frames       != old_h)

    def _get_q(self, state: Tuple) -> List[float]:
        if state not in self._q:
            self._q[state] = [0.0] * N_ACTIONS
        return self._q[state]

    @staticmethod
    def _miss_rate(window: deque) -> float:
        if not window:
            return 0.0
        return sum(1 for x in window if not x) / len(window)
