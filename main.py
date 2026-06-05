# -*- coding: utf-8 -*-
"""
主程序入口 — 游戏主循环
整合视觉感知、象棋逻辑、AI博弈、康复评估、可视化五大模块

运行流程：
  1. 初始化所有模块
  2. 主循环每帧：读取摄像头 → 手势识别 → 手势状态机处理 → AI走棋 → 渲染
  3. 按 R 重新开始，ESC/Q 退出
"""

import sys
import threading
import time
from typing import List, Optional, Tuple

import pygame

import config
from chess_game import ChessGame, RED, BLACK
from vision import VisionModule, GestureState
from assessment import AssessmentModule
from ai_engine import AIEngine
from gesture_rl import GestureRL
from ui import GameUI


class RehabChessApp:
    def __init__(self):
        self.game       = ChessGame()
        self.vision     = VisionModule()
        self.assessment = AssessmentModule()
        self.ai         = AIEngine()
        self.ui         = GameUI()
        # 手势强化学习控制器：初始阈值从 config 读取
        self.gesture_rl = GestureRL(
            init_pinch   = config.GESTURE_PINCH_THRESHOLD,
            init_release = config.GESTURE_RELEASE_THRESHOLD,
            init_hold    = config.GESTURE_MIN_HOLD_FRAMES,
        )

        # 交互状态
        self.selected_pos:  Optional[Tuple[int, int]] = None
        self.legal_moves:   List[Tuple[int, int]]     = []
        self.ai_thinking:   bool  = False
        self.ai_move_ready: Optional[Tuple] = None
        self._ai_thread:    Optional[threading.Thread] = None

        # 手势防抖：RELEASING 只触发一次
        self._release_handled = False

        # 停留超时：IDLE 状态手在棋盘同一格停留 3 秒未操作，触发阈值下调
        self._dwell_start:    Optional[float]        = None
        self._dwell_last_pos: Optional[Tuple[int,int]] = None
        self._DWELL_TIMEOUT = 3.0

        # 悔棋剪刀手计时
        self._undo_hover_start: Optional[float] = None
        self._UNDO_HOLD_SECS = 3.0  # 剪刀手保持3秒触发
        self._scissors_enabled = False  # 仅玩家走棋成功后才启用剪刀手检测

        # 消息
        self._pending_message: Optional[str] = None

    # ── 主循环 ─────────────────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            # ── Pygame 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        self._restart()
                    elif event.key == pygame.K_z:
                        self._undo()

            # ── 摄像头 + 手势识别
            # RL 实时阈值注入：每帧将 GestureRL 最新阈值同步给 VisionModule
            rl_stats = self.gesture_rl.get_stats()
            self.vision.set_thresholds(
                rl_stats.pinch_threshold,
                rl_stats.release_threshold,
                rl_stats.hold_frames,
            )
            gesture, cam_frame = self.vision.update()

            # ── 手势交互逻辑（仅玩家回合）
            if not self.game.game_over and self.game.current_turn == RED:
                self._handle_gesture(gesture)

            # ── AI 走棋（异步）
            self._handle_ai_turn()

            # ── 渲染
            stats = self.assessment.get_stats()
            self.ui.scissors_enabled = self._scissors_enabled
            self.ui.render(
                game         = self.game,
                stats        = stats,
                gesture      = gesture,
                cam_frame    = cam_frame,
                selected_pos = self.selected_pos,
                legal_moves  = self.legal_moves,
                ai_thinking  = self.ai_thinking,
                message      = self._pending_message,
                rl_stats     = rl_stats,
            )
            self._pending_message = None

        self.vision.release()
        self.ui.quit()

    # ── 手势交互 ───────────────────────────────────────────────────────────────

    def _handle_gesture(self, gesture):
        if not gesture.hand_detected:
            self._undo_hover_start = None
            self.ui.undo_hover_progress = 0.0
            self._dwell_start    = None
            self._dwell_last_pos = None
            return

        # ── 剪刀手悔棋 ────────────────────────────────────────────────────
        if self._scissors_enabled and gesture.is_scissors:
            if self._undo_hover_start is None:
                self._undo_hover_start = time.time()
            elapsed = time.time() - self._undo_hover_start
            self.ui.undo_hover_progress = min(elapsed / self._UNDO_HOLD_SECS, 1.0)
            if elapsed >= self._UNDO_HOLD_SECS:
                self._undo_hover_start = None
                self.ui.undo_hover_progress = 0.0
                self._undo()
            return
        else:
            self._undo_hover_start = None
            self.ui.undo_hover_progress = 0.0

        # ── 棋盘手势交互 ────────────────────────────────────────────────────
        board_pos = self.ui.camera_to_board(gesture.norm_x, gesture.norm_y)

        if gesture.state == GestureState.GRABBING:
            self._release_handled = False
            self._dwell_start    = None
            self._dwell_last_pos = None
            if board_pos is not None:
                piece = self.game.board[board_pos[0]][board_pos[1]]
                from chess_game import _color
                has_own_piece = (piece != 0 and _color(piece) == RED)
                if has_own_piece and self.selected_pos != board_pos:
                    # 握紧成功：起点有效，有己方棋子
                    self.selected_pos = board_pos
                    self.legal_moves  = self.game.get_legal_moves(*board_pos)
                    self.assessment.start_move_timer()
                    self.gesture_rl.record_grab_success()
                    self._pending_message = f"选中棋子: 行{board_pos[0]+1} 列{board_pos[1]+1}"
                elif not has_own_piece and self.selected_pos is None:
                    # 握紧失误：起点无效，该位置无己方棋子
                    # RL 将倾向于调宽捏合阈值或增加保持帧数
                    self.gesture_rl.record_grab_miss()

        elif gesture.state == GestureState.RELEASING and not self._release_handled:
            self._release_handled = True
            self._dwell_start    = None
            self._dwell_last_pos = None
            self.assessment.record_gesture_attempt()

            if self.selected_pos is not None and board_pos is not None:
                self._try_place(board_pos)
            else:
                # 松开失误：无选中子 或 board_pos 为 None（棋盘外）
                # RL 将倾向于调宽松开阈值
                self.gesture_rl.record_release_miss()
                if self.selected_pos is not None:
                    self._deselect()
                self.assessment.record_illegal_attempt()
                self._pending_message = "松开位置无效，取消选子"

        elif gesture.state == GestureState.IDLE and board_pos is not None:
            # 停留超时检测：手在棋盘同一格 IDLE 超过 3 秒未完成操作
            if self._dwell_last_pos == board_pos:
                if self._dwell_start is None:
                    self._dwell_start = time.time()
                elif time.time() - self._dwell_start >= self._DWELL_TIMEOUT:
                    self.gesture_rl.record_dwell_timeout()
                    self._dwell_start = None  # 触发后重置，避免连续触发
                    self._pending_message = "停留超时，阈值已自动降低"
            else:
                self._dwell_last_pos = board_pos
                self._dwell_start    = None
        else:
            self._dwell_start    = None
            self._dwell_last_pos = None

    def _try_place(self, target: Tuple[int, int]):
        if target == self.selected_pos:
            self._deselect()
            return

        if target in self.legal_moves:
            fr, fc = self.selected_pos
            tr, tc = target
            captured = self.game.board[tr][tc] != 0
            success = self.game.make_move(fr, fc, tr, tc)
            if success:
                # 松开成功：落点合法，完整动作完成
                self.gesture_rl.record_release_success()
                self.assessment.record_move(captured=captured)
                cap_msg = " (吃子)" if captured else ""
                self._pending_message = f"走棋: {self.selected_pos} → {target}{cap_msg}"
                self._deselect()
                self._scissors_enabled = True
                if self.game.game_over:
                    self.assessment.record_game_result(
                        player_won=(self.game.winner == RED)
                    )
            else:
                # 内部错误（理论上不应发生）
                self.gesture_rl.record_release_miss()
                self.assessment.record_illegal_attempt()
                self._pending_message = "走棋失败（内部错误）"
                self._deselect()
        else:
            # 松开失误：落点不在合法走法内
            # RL 将观测到 release_miss，倾向于调整松开阈值
            self.gesture_rl.record_release_miss()
            self.assessment.record_illegal_attempt()
            self._pending_message = f"非法走法：{self.selected_pos} → {target}"

    def _deselect(self):
        self.selected_pos = None
        self.legal_moves  = []

    # ── AI 走棋 ────────────────────────────────────────────────────────────────

    def _handle_ai_turn(self):
        # 等待 AI 线程结果
        if self.ai_move_ready is not None:
            move = self.ai_move_ready
            self.ai_move_ready = None
            self.ai_thinking   = False
            if move is not None and not self.game.game_over:
                fr, fc, tr, tc = move
                captured = self.game.board[tr][tc] != 0
                self.game.make_move(fr, fc, tr, tc)
                cap_msg = " (AI吃子)" if captured else ""
                self._pending_message = f"AI走棋: ({fr},{fc}) → ({tr},{tc}){cap_msg}"
                # 游戏结束时触发局末 Q-Learning 奖励
                if self.game.game_over:
                    self.assessment.record_game_result(
                        player_won=(self.game.winner == RED)
                    )
            return

        # 触发 AI 走棋
        if (
            not self.game.game_over
            and self.game.current_turn == BLACK
            and not self.ai_thinking
            and self._ai_thread is None
        ):
            self.ai_thinking = True
            stats_snap = self.assessment.get_stats()
            skill_score = stats_snap.skill_score  # 实时棋力分 0~100
            game_copy = self.game.board_copy()

            def _ai_work():
                time.sleep(config.AI_THINK_DELAY)
                # 注释：传入实时棋力分，AI据此计算放水率和搜索深度
                move = self.ai.get_move(
                    game_copy, skill_score,
                    stats=self.assessment.get_stats(),
                )
                self.ai_move_ready = move
                self._ai_thread = None

            self._ai_thread = threading.Thread(target=_ai_work, daemon=True)
            self._ai_thread.start()

    # ── 重新开始 ───────────────────────────────────────────────────────────────

    def _undo(self):
        if self.game.game_over:
            return
        # AI 还在思考时不允许悔棋，避免状态混乱
        if self.ai_thinking or self._ai_thread is not None:
            self._pending_message = "AI 思考中，无法悔棋"
            return
        ok = self.game.undo_two_moves()
        if ok:
            self._deselect()
            self._scissors_enabled = False  # 悔棋后需再次走棋才能悔棋
            self.assessment.record_illegal_attempt()
            self._pending_message = "悔棋成功"
        else:
            self._pending_message = "无棋可悔"

    def _restart(self):
        self.game       = ChessGame()
        self.assessment.reset()
        # RL 重置：保留已学到的 Q 表，只重置事件窗口和连续成功计数
        # 这样跨局累积的阈值经验不会丢失
        self.gesture_rl._grab_window.clear()
        self.gesture_rl._release_window.clear()
        self.gesture_rl._all_window.clear()
        self.gesture_rl._success_streak = 0
        self._deselect()
        self.ai_thinking    = False
        self.ai_move_ready  = None
        self._ai_thread     = None
        self._release_handled = False
        self._undo_hover_start = None
        self.ui.undo_hover_progress = 0.0
        self._scissors_enabled = False
        self._dwell_start    = None
        self._dwell_last_pos = None
        self._pending_message = "新对局开始！"


def main():
    if not pygame.get_init():
        pygame.init()
    app = RehabChessApp()

    if not app.vision.is_camera_open:
        print("[警告] 摄像头未能打开，将以无摄像头模式运行")

    app.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
