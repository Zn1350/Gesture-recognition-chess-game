# -*- coding: utf-8 -*-
"""
可视化模块 — Pygame 游戏主界面
布局：左侧摄像头预览 | 右上象棋对弈区 | 右下+左下康复数据HUD
"""

from __future__ import annotations
import os
from typing import List, Optional, Tuple

import pygame
import numpy as np
import cv2

import config
from chess_game import (
    ChessGame, RED, BLACK, EMPTY, PIECE_CHARS,
    R_KING, R_GUARD, R_BISHOP, R_ROOK, R_KNIGHT, R_CANNON, R_PAWN,
)
from assessment import RehabStats
from vision import GestureState, GestureResult
from gesture_rl import GestureRLStats


def _load_font(size: int) -> pygame.font.Font:
    """按优先级尝试加载中文字体"""
    for path in config.FONT_PATHS:
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except Exception:
                pass
    return pygame.font.SysFont("arial", size)


class GameUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        self.clock = pygame.time.Clock()

        # 字体
        self.font_lg  = _load_font(config.FONT_SIZE_LARGE)
        self.font_md  = _load_font(config.FONT_SIZE_MEDIUM)
        self.font_sm  = _load_font(config.FONT_SIZE_SMALL)
        self.font_pc  = _load_font(config.FONT_SIZE_PIECE)

        # 棋盘绘制区域
        bx, by, bw, bh = config.BOARD_DISPLAY_RECT
        self._board_x = bx
        self._board_y = by
        self._board_w = bw
        self._board_h = bh
        self._board_padding = config.BOARD_PADDING
        self._board_v_padding_top = config.BOARD_VERTICAL_PADDING_TOP
        self._board_v_padding_bottom = config.BOARD_VERTICAL_PADDING_BOTTOM
        inner_w = bw - 2 * self._board_padding
        inner_h = bh - self._board_v_padding_top - self._board_v_padding_bottom
        self._cell_w  = inner_w / (config.BOARD_COLS - 1)
        self._cell_h  = inner_h / (config.BOARD_ROWS - 1)

        # 摄像头显示区域
        cx, cy, cw, ch = config.CAMERA_DISPLAY_RECT
        self._cam_x = cx
        self._cam_y = cy
        self._cam_w = cw
        self._cam_h = ch

        # HUD 区域
        hx, hy, hw, hh = config.HUD_RECT
        self._hud_x = hx
        self._hud_y = hy
        self._hud_w = hw
        self._hud_h = hh

        # 摄像头帧 surface
        self._cam_surf: Optional[pygame.Surface] = None

        # 消息日志（最多显示5条）
        self._messages: List[str] = []

        # 悔棋提示进度：用于内部剪刀手计时反馈，界面不再展示按钮
        self.undo_hover_progress: float = 0.0
        self.undo_hold_secs: float = 3.0
        self.scissors_enabled: bool = False  # 由 main.py 同步，控制图标是否显示

    # ── 主渲染入口 ────────────────────────────────────────────────────────────

    def render(
        self,
        game: ChessGame,
        stats: RehabStats,
        gesture: GestureResult,
        cam_frame,
        selected_pos: Optional[Tuple[int, int]],
        legal_moves: List[Tuple[int, int]],
        ai_thinking: bool,
        message: Optional[str] = None,
        rl_stats: Optional[GestureRLStats] = None,
    ):
        if message:
            self._push_message(message)

        self.screen.fill(config.COLOR_BG)

        self._draw_camera(cam_frame, gesture)
        self._draw_board(game, selected_pos, legal_moves)
        self._draw_hud(stats, game, ai_thinking, gesture, rl_stats)
        self._draw_dividers()
        self._draw_scissors_overlay(gesture)
        self._draw_gesture_cursor(gesture)

        if ai_thinking:
            self._draw_thinking_overlay()

        pygame.display.flip()
        self.clock.tick(config.CAMERA_FPS)

    # ── 摄像头预览区 ──────────────────────────────────────────────────────────

    def _draw_camera(self, frame, gesture: GestureResult):
        # 背景框
        pygame.draw.rect(self.screen, config.COLOR_DARK_GRAY,
                         (self._cam_x, self._cam_y, self._cam_w, self._cam_h))

        if frame is not None:
            # BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 保持摄像头原始宽高比，避免纵向拉伸
            src_h, src_w = rgb.shape[:2]
            src_ratio = src_w / src_h
            dst_ratio = self._cam_w / self._cam_h
            if dst_ratio > src_ratio:
                out_h = self._cam_h
                out_w = int(src_ratio * out_h)
            else:
                out_w = self._cam_w
                out_h = int(out_w / src_ratio)
            rgb_resized = cv2.resize(rgb, (out_w, out_h))
            surf = pygame.surfarray.make_surface(np.transpose(rgb_resized, (1, 0, 2)))
            offset_x = self._cam_x + (self._cam_w - out_w) // 2
            offset_y = self._cam_y + (self._cam_h - out_h) // 2
            self.screen.blit(surf, (offset_x, offset_y))
        else:
            # 无摄像头时显示占位文字
            txt = self.font_md.render("[ 摄像头未连接 ]", True, config.COLOR_GRAY)
            self.screen.blit(txt, (self._cam_x + 200, self._cam_y + 220))

        # 手势状态指示条
        state_colors = {
            GestureState.IDLE:      config.COLOR_GRAY,
            GestureState.GRABBING:  config.COLOR_GREEN,
            GestureState.RELEASING: config.COLOR_YELLOW,
        }
        sc = state_colors.get(gesture.state, config.COLOR_GRAY)
        state_names = {
            GestureState.IDLE:      "待机",
            GestureState.GRABBING:  "抓取",
            GestureState.RELEASING: "释放",
        }
        pygame.draw.rect(self.screen, sc,
                         (self._cam_x, self._cam_y + self._cam_h - 24, self._cam_w, 24))
        lbl = self.font_sm.render(
            f"手势: {state_names.get(gesture.state, '')}  "
            f"{'已检测到手部' if gesture.hand_detected else '未检测到手部'}",
            True, (0, 0, 0)
        )
        self.screen.blit(lbl, (self._cam_x + 8, self._cam_y + self._cam_h - 20))

        # 标题
        t = self.font_sm.render("摄像头 / 手势识别", True, config.COLOR_CYAN)
        self.screen.blit(t, (self._cam_x + 4, self._cam_y + 2))

    # ── 象棋棋盘区 ────────────────────────────────────────────────────────────

    def _board_to_screen(self, r: int, c: int) -> Tuple[int, int]:
        """棋盘格坐标 → 屏幕像素坐标"""
        x = self._board_x + self._board_padding + c * self._cell_w
        y = self._board_y + self._board_v_padding_top + r * self._cell_h
        return (int(x), int(y))

    def _draw_board(
        self,
        game: ChessGame,
        selected_pos: Optional[Tuple[int, int]],
        legal_moves: List[Tuple[int, int]],
    ):
        # 棋盘背景
        pygame.draw.rect(
            self.screen, config.BOARD_BG_COLOR,
            (self._board_x - config.PIECE_RADIUS,
             self._board_y - config.PIECE_RADIUS,
             self._board_w + config.PIECE_RADIUS * 2,
             self._board_h + config.PIECE_RADIUS * 2)
        )

        # 棋盘线条
        for r in range(config.BOARD_ROWS):
            sx, sy = self._board_to_screen(r, 0)
            ex, ey = self._board_to_screen(r, config.BOARD_COLS - 1)
            pygame.draw.line(self.screen, config.BOARD_LINE_COLOR, (sx, sy), (ex, ey), 1)

        for c in range(config.BOARD_COLS):
            # 上半段（黑方阵地）
            sx, sy = self._board_to_screen(0, c)
            ex, ey = self._board_to_screen(4, c)
            pygame.draw.line(self.screen, config.BOARD_LINE_COLOR, (sx, sy), (ex, ey), 1)
            # 下半段（红方阵地）
            sx, sy = self._board_to_screen(5, c)
            ex, ey = self._board_to_screen(9, c)
            pygame.draw.line(self.screen, config.BOARD_LINE_COLOR, (sx, sy), (ex, ey), 1)

        # 楚河 / 汉界：分别置于河界两侧
        hx1, hy1 = self._board_to_screen(4, 0)
        hx2, hy2 = self._board_to_screen(5, 0)
        mid_y = (hy1 + hy2) // 2
        river_r = pygame.Rect(hx1 - config.PIECE_RADIUS,
                              mid_y - 16,
                              self._board_w + config.PIECE_RADIUS * 2, 32)
        pygame.draw.rect(self.screen, config.BOARD_BG_COLOR, river_r)
        left_txt = self.font_md.render("楚河", True, config.BOARD_LINE_COLOR)
        right_txt = self.font_md.render("汉界", True, config.BOARD_LINE_COLOR)
        self.screen.blit(left_txt, (hx1 + self._board_w * 0.15, mid_y - 12))
        self.screen.blit(right_txt, (hx1 + self._board_w * 0.65, mid_y - 12))

        # 九宫斜线（双方各一个）
        for palace_rows, palace_cols in [((0, 2), (3, 5)), ((7, 9), (3, 5))]:
            tl = self._board_to_screen(palace_rows[0], palace_cols[0])
            br = self._board_to_screen(palace_rows[1], palace_cols[1])
            tr = self._board_to_screen(palace_rows[0], palace_cols[1])
            bl = self._board_to_screen(palace_rows[1], palace_cols[0])
            pygame.draw.line(self.screen, config.BOARD_LINE_COLOR, tl, br, 1)
            pygame.draw.line(self.screen, config.BOARD_LINE_COLOR, tr, bl, 1)

        # 可落子高亮
        for (mr, mc) in legal_moves:
            px, py = self._board_to_screen(mr, mc)
            pygame.draw.circle(self.screen, config.LEGAL_COLOR, (px, py),
                               config.PIECE_RADIUS + 4, 3)

        # 选中高亮
        if selected_pos:
            sr, sc_col = selected_pos
            px, py = self._board_to_screen(sr, sc_col)
            pygame.draw.circle(self.screen, config.SELECT_COLOR, (px, py),
                               config.PIECE_RADIUS + 5, 3)

        # 绘制棋子
        for r in range(config.BOARD_ROWS):
            for c in range(config.BOARD_COLS):
                piece = game.board[r][c]
                if piece != EMPTY:
                    self._draw_piece(r, c, piece, selected=(selected_pos == (r, c)))

        # 列标 abcdefghi 行标 1-10（辅助定位）
        for c in range(config.BOARD_COLS):
            px, py = self._board_to_screen(9, c)
            lbl = self.font_sm.render(str(c + 1), True, config.BOARD_LINE_COLOR)
            self.screen.blit(lbl, (px - 5, py + config.PIECE_RADIUS + 2))

    def _draw_piece(self, r: int, c: int, piece: int, selected: bool = False):
        px, py = self._board_to_screen(r, c)
        color  = config.RED_COLOR if piece > 0 else config.BLACK_COLOR
        border = config.SELECT_COLOR if selected else (220, 180, 100)

        pygame.draw.circle(self.screen, border, (px, py), config.PIECE_RADIUS + 2)
        pygame.draw.circle(self.screen, color,  (px, py), config.PIECE_RADIUS)

        char = PIECE_CHARS.get(piece, "?")
        txt  = self.font_pc.render(char, True, config.COLOR_WHITE)
        tr   = txt.get_rect(center=(px, py))
        self.screen.blit(txt, tr)

    # ── 康复数据 HUD ──────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        stats: RehabStats,
        game: ChessGame,
        ai_thinking: bool,
        gesture: GestureResult,
        rl_stats: Optional[GestureRLStats] = None,
    ):
        hx, hy = self._hud_x, self._hud_y
        hw, hh = self._hud_w, self._hud_h

        pygame.draw.rect(self.screen, config.COLOR_DARK_GRAY, (hx, hy, hw, hh))
        pygame.draw.line(self.screen, config.COLOR_GRAY, (hx, hy), (hx + hw, hy), 1)

        # ── 标题
        title = self.font_md.render("康复训练数据", True, config.COLOR_CYAN)
        self.screen.blit(title, (hx + 10, hy + 6))

        # ── 实时棋力分（0~100 进度条）
        score = stats.skill_score
        bar_total = 320
        bar_fill  = int(score / 100 * bar_total)
        score_color = (
            config.COLOR_RED    if score < 20 else
            config.COLOR_ORANGE if score < 40 else
            config.COLOR_YELLOW if score < 60 else
            config.COLOR_GREEN  if score < 80 else
            config.COLOR_CYAN
        )
        pygame.draw.rect(self.screen, (50, 50, 60), (hx + 200, hy + 4, bar_total, 18), border_radius=4)
        pygame.draw.rect(self.screen, score_color,  (hx + 200, hy + 4, bar_fill,  18), border_radius=4)
        score_lbl = self.font_md.render(f"棋力分: {score:.0f}/100  {stats.skill_label}", True, score_color)
        self.screen.blit(score_lbl, (hx + 200, hy + 24))

        # ── 数据列（两列布局）
        col1_x = hx + 10
        col2_x = hx + 320
        col3_x = hx + 640
        col4_x = hx + 960
        row1_y = hy + 48
        row2_y = hy + 72
        row3_y = hy + 96
        row4_y = hy + 120
        row5_y = hy + 144
        row6_y = hy + 168

        def draw_kv(x, y, key, val, vcolor=config.COLOR_WHITE):
            k = self.font_sm.render(key, True, config.COLOR_GRAY)
            v = self.font_md.render(str(val), True, vcolor)
            self.screen.blit(k, (x, y))
            self.screen.blit(v, (x + 80, y - 2))

        draw_kv(col1_x, row1_y, "总步数:", stats.total_moves, config.COLOR_WHITE)
        draw_kv(col1_x, row2_y, "失误次:", stats.illegal_attempts, config.COLOR_RED)
        draw_kv(col1_x, row3_y, "吃子数:", stats.captures, config.COLOR_YELLOW)
        draw_kv(col1_x, row4_y, "失误率:", f"{stats.error_rate*100:.1f}%",
                config.COLOR_RED if stats.error_rate > 0.2 else config.COLOR_GREEN)
        draw_kv(col1_x, row5_y, "平均步时:", f"{stats.avg_move_time:.1f}s")
        draw_kv(col1_x, row6_y, "局时:", self._fmt_time(stats.session_seconds))

        # 手势准确率
        draw_kv(col2_x, row1_y, "手势精准:", f"{stats.gesture_accuracy*100:.0f}%",
                config.COLOR_GREEN if stats.gesture_accuracy > 0.8 else config.COLOR_ORANGE)
        draw_kv(col2_x, row2_y, "手势触发:", stats.gesture_attempts)

        # 当前轮次
        turn_color = config.RED_COLOR if game.current_turn == RED else config.BLACK_COLOR
        turn_txt = "红方" if game.current_turn == RED else "黑方(AI)"
        draw_kv(col2_x, row3_y, "当前轮:", turn_txt, turn_color)

        # AI状态
        ai_status = "思考中..." if ai_thinking else "等待中"
        draw_kv(col2_x, row4_y, "AI状态:", ai_status,
                config.COLOR_YELLOW if ai_thinking else config.COLOR_GRAY)

        # AI放水率（Q-Learning 自适应）
        br = stats.ai_blunder_rate
        blunder_color = config.COLOR_GREEN if br > 0.4 else \
                        config.COLOR_YELLOW if br > 0.2 else config.COLOR_RED
        draw_kv(col2_x, row5_y, "AI放水:", f"{br*100:.0f}%", blunder_color)

        # Q-Learning 奖励值（上一步）
        reward = stats.last_ql_reward
        r_color = config.COLOR_GREEN if reward > 0 else \
                  config.COLOR_RED   if reward < 0 else config.COLOR_GRAY
        draw_kv(col2_x, row6_y, "QL奖励:", f"{reward:+.1f}", r_color)

        # 捏合比率可视化
        self._draw_pinch_bar(col3_x, row1_y, gesture.pinch_ratio)

        # 按键提示
        hints = self.font_sm.render("Z/剪刀手3秒=悔棋   R=重新开始   ESC=退出", True, config.COLOR_GRAY)
        self.screen.blit(hints, (col3_x, row6_y))

        # 消息日志
        for i, msg in enumerate(self._messages[-5:]):
            c_ = (min(255, 180), min(255, 180 - i * 20), min(255, 100 - i * 10))
            m = self.font_sm.render(msg, True, c_)
            self.screen.blit(m, (col3_x, row2_y + i * 22))

        # ── 手势 RL 面板（col4 区域）────────────────────────────────────────
        if rl_stats is not None and not game.game_over:
            self._draw_gesture_rl_panel(col4_x, self._hud_y, rl_stats)

        # 游戏结束提示
        if game.game_over:
            winner_txt = "红方胜！玩家获胜！" if game.winner == RED else "黑方(AI)胜！"
            w_color = config.COLOR_YELLOW
            w_surf = self.font_lg.render(f"对局结束: {winner_txt}", True, w_color)
            self.screen.blit(w_surf, (col4_x - 100, row3_y))
            hint = self.font_sm.render("按 R 重新开始", True, config.COLOR_GRAY)
            self.screen.blit(hint, (col4_x - 60, row4_y))

    def _draw_gesture_rl_panel(self, x: int, y: int, rl: GestureRLStats):
        """
        手势强化学习状态面板
        展示实时阈值、两类失误率、RL 奖励、连续成功、最近动作
        """
        title = self.font_sm.render("手势RL自适应", True, config.COLOR_CYAN)
        self.screen.blit(title, (x, y + 6))

        def kv(dy, key, val, vc=config.COLOR_WHITE):
            k = self.font_sm.render(key, True, config.COLOR_GRAY)
            v = self.font_sm.render(str(val), True, vc)
            self.screen.blit(k, (x, y + dy))
            self.screen.blit(v, (x + 78, y + dy))

        # 当前阈值
        kv(26, "捏合阈:", f"{rl.pinch_threshold:.3f}", config.COLOR_YELLOW)
        kv(44, "松开阈:", f"{rl.release_threshold:.3f}", config.COLOR_YELLOW)
        kv(62, "保持帧:", f"{rl.hold_frames}帧", config.COLOR_YELLOW)

        # 两类失误率（核心 RL 输入信号）
        gm_c = config.COLOR_RED if rl.grab_miss_rate > 0.3 else \
               config.COLOR_ORANGE if rl.grab_miss_rate > 0.1 else config.COLOR_GREEN
        rm_c = config.COLOR_RED if rl.release_miss_rate > 0.3 else \
               config.COLOR_ORANGE if rl.release_miss_rate > 0.1 else config.COLOR_GREEN
        kv(82, "握失误:", f"{rl.grab_miss_rate*100:.0f}%", gm_c)
        kv(100, "松失误:", f"{rl.release_miss_rate*100:.0f}%", rm_c)

        # RL 奖励和连续成功
        r_c = config.COLOR_GREEN if rl.last_reward > 0 else \
              config.COLOR_RED if rl.last_reward < 0 else config.COLOR_GRAY
        kv(118, "RL奖励:", f"{rl.last_reward:+.1f}", r_c)
        streak_c = config.COLOR_CYAN if rl.success_streak >= 8 else config.COLOR_WHITE
        kv(136, "连续成功:", f"{rl.success_streak}次", streak_c)

        # 最近执行的 RL 动作（阈值调整行为）
        kv(154, "上次调整:", rl.rl_action_name, config.COLOR_ORANGE)

        # 累计调整次数
        kv(172, "调整次数:", f"{rl.total_adjustments}次", config.COLOR_GRAY)

    def _draw_pinch_bar(self, x: int, y: int, ratio: float):
        """捏合程度进度条"""
        label = self.font_sm.render("捏合程度:", True, config.COLOR_GRAY)
        self.screen.blit(label, (x, y))
        bar_x, bar_y = x, y + 18
        bar_w, bar_h = 200, 14
        pygame.draw.rect(self.screen, config.COLOR_DARK_GRAY, (bar_x, bar_y, bar_w, bar_h))
        # ratio 越小 = 捏合越紧，填充越多
        fill = int((1.0 - min(ratio, 0.3) / 0.3) * bar_w)
        fill_color = config.COLOR_GREEN if ratio < config.GESTURE_PINCH_THRESHOLD \
                     else config.COLOR_ORANGE
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill, bar_h))
        pygame.draw.rect(self.screen, config.COLOR_GRAY, (bar_x, bar_y, bar_w, bar_h), 1)
        # 阈值线
        thresh_x = bar_x + int((1.0 - config.GESTURE_PINCH_THRESHOLD / 0.3) * bar_w)
        pygame.draw.line(self.screen, config.COLOR_RED,
                         (thresh_x, bar_y), (thresh_x, bar_y + bar_h), 2)

    def _draw_dividers(self):
        """区域分割线"""
        # 顶部区域底部分割线
        pygame.draw.line(self.screen, config.COLOR_GRAY,
                         (0, self._hud_y), (config.WINDOW_WIDTH, self._hud_y), 1)
        # 摄像头区域右侧边界
        pygame.draw.line(self.screen, config.COLOR_GRAY,
                         (self._cam_x + self._cam_w, 0), (self._cam_x + self._cam_w, self._hud_y), 1)

    def _draw_scissors_overlay(self, gesture: GestureResult):
        """
        剪刀手悔棋进度覆盖层
        - 未激活：摄像头区右下角显示小图标提示
        - 激活中：摄像头区中央大圆形进度环 + 倒计时数字
        """
        import math

        progress = self.undo_hover_progress  # 0.0 ~ 1.0
        is_scissors = getattr(gesture, 'is_scissors', False)

        # ── 静止提示：右下角小图标（仅启用时显示） ─────────────────────────
        if not self.scissors_enabled and self.undo_hover_progress <= 0:
            return
        hint_x = self._cam_x + self._cam_w - 110
        hint_y = self._cam_y + self._cam_h - 54
        hint_bg = pygame.Surface((104, 28), pygame.SRCALPHA)
        hint_bg.fill((0, 0, 0, 120))
        self.screen.blit(hint_bg, (hint_x, hint_y))
        hint_color = (80, 220, 80) if is_scissors else (140, 140, 140)
        hint_txt = self.font_sm.render("✌ 悔棋(3s)", True, hint_color)
        self.screen.blit(hint_txt, (hint_x + 4, hint_y + 4))

        if progress <= 0:
            return

        # ── 激活中：摄像头区中央大进度环 ────────────────────────────────────
        cx = self._cam_x + self._cam_w // 2
        cy = self._cam_y + self._cam_h // 2
        R = 70  # 外圆半径

        # 半透明遮罩
        mask = pygame.Surface((self._cam_w, self._cam_h), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 100))
        self.screen.blit(mask, (self._cam_x, self._cam_y))

        # 灰色底圆环
        pygame.draw.circle(self.screen, (60, 60, 60), (cx, cy), R, 6)

        # 绿色进度弧：用多段线模拟（pygame.draw.arc 不支持粗线）
        arc_steps = max(6, int(90 * progress))
        start_angle = math.pi / 2           # 12点方向
        sweep = -2 * math.pi * progress     # 顺时针
        prev_pt = None
        for i in range(arc_steps + 1):
            t = i / arc_steps
            a = start_angle + sweep * t
            pt = (
                int(cx + R * math.cos(a)),
                int(cy - R * math.sin(a)),
            )
            if prev_pt is not None:
                # 绿色渐变：起点草绿，终点亮绿
                g = int(160 + 95 * t)
                pygame.draw.line(self.screen, (30, g, 60), prev_pt, pt, 7)
            prev_pt = pt

        # 进度环端点亮点
        end_a = start_angle + sweep
        end_pt = (int(cx + R * math.cos(end_a)), int(cy - R * math.sin(end_a)))
        pygame.draw.circle(self.screen, (120, 255, 120), end_pt, 6)

        # 倒计时数字（中央大字）
        remaining = max(0.0, self.undo_hold_secs * (1.0 - progress))
        num_surf = self.font_lg.render(f"{remaining:.1f}", True, (120, 255, 120))
        self.screen.blit(num_surf, num_surf.get_rect(center=(cx, cy - 12)))

        # "悔棋" 小字
        sub_surf = self.font_sm.render("✌ 保持剪刀手", True, (180, 255, 180))
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(cx, cy + 18)))

    def _draw_gesture_cursor(self, gesture: GestureResult):
        """在棋盘上绘制手势光标，让玩家直观确认手的映射位置"""
        if not gesture.hand_detected:
            return
        px, py = self.get_gesture_screen_pos(gesture)
        color = config.COLOR_GREEN  if gesture.state == GestureState.GRABBING  else \
                config.COLOR_YELLOW if gesture.state == GestureState.RELEASING else \
                config.COLOR_WHITE
        # 十字准星
        r = 14
        pygame.draw.line(self.screen, color, (px - r, py), (px + r, py), 2)
        pygame.draw.line(self.screen, color, (px, py - r), (px, py + r), 2)
        pygame.draw.circle(self.screen, color, (px, py), r, 2)

    def _draw_thinking_overlay(self):
        """AI思考时遮罩提示"""
        s = pygame.Surface((self._board_w + 60, 40), pygame.SRCALPHA)
        s.fill((0, 0, 0, 140))
        self.screen.blit(s, (self._board_x - 10, self._board_y + self._board_h // 2 - 20))
        txt = self.font_md.render("AI 正在思考...", True, config.COLOR_YELLOW)
        self.screen.blit(txt, (self._board_x + self._board_w // 2 - 70,
                                self._board_y + self._board_h // 2 - 14))

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _board_rect(self) -> Tuple[int, int, int, int]:
        """返回整个显示棋盘的外框，包含左右和上下 55px 空白。"""
        return (self._board_x, self._board_y, self._board_w, self._board_h)

    def _grid_rect(self) -> Tuple[int, int, int, int]:
        """返回棋盘内部实际网格可点选区域，不含左右/上下内部空白。"""
        left = self._board_x + self._board_padding
        top = self._board_y + self._board_v_padding_top
        width = self._board_w - 2 * self._board_padding
        height = self._board_h - self._board_v_padding_top - self._board_v_padding_bottom
        return (left, top, width, height)

    def screen_to_board(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        """屏幕像素坐标 → 最近棋盘格坐标。

        这里使用的是“原始棋盘显示区域 + 35px 内边距”的整体棋盘范围。
        如果点位于该区域之外，则返回 None；如果点落在空白边距中则钳制到最近可选格。
        """
        board_left, board_top, board_w, board_h = self._board_rect()
        board_right = board_left + board_w
        board_bottom = board_top + board_h

        if px < board_left or px > board_right or py < board_top or py > board_bottom:
            return None

        grid_left, grid_top, grid_w, grid_h = self._grid_rect()
        grid_right = grid_left + grid_w
        grid_bottom = grid_top + grid_h

        clamped_x = min(max(px, grid_left), grid_right)
        clamped_y = min(max(py, grid_top), grid_bottom)

        col = round((clamped_x - grid_left) / grid_w * (config.BOARD_COLS - 1))
        row = round((clamped_y - grid_top) / grid_h * (config.BOARD_ROWS - 1))
        col = max(0, min(config.BOARD_COLS - 1, col))
        row = max(0, min(config.BOARD_ROWS - 1, row))
        return (row, col)

    def camera_to_board(self, cam_norm_x: float, cam_norm_y: float) -> Optional[Tuple[int, int]]:
        """摄像头归一化坐标 → 棋盘格坐标。

        将手势归一化坐标映射到整个带 35px 内边距的棋盘区域，
        再转换到棋盘网格坐标。
        """
        x = max(0.0, min(1.0, cam_norm_x))
        y = max(0.0, min(1.0, cam_norm_y))

        board_left, board_top, board_w, board_h = self._board_rect()
        screen_x = board_left + int(round(x * board_w))
        screen_y = board_top  + int(round(y * board_h))
        return self.screen_to_board(screen_x, screen_y)

    def get_gesture_screen_pos(self, gesture: GestureResult) -> Tuple[int, int]:
        """手势归一化坐标在显示棋盘区域的屏幕坐标。"""
        x = max(0.0, min(1.0, gesture.norm_x))
        y = max(0.0, min(1.0, gesture.norm_y))
        board_left, board_top, board_w, board_h = self._board_rect()
        px = board_left + int(round(x * board_w))
        py = board_top  + int(round(y * board_h))
        return (px, py)

    def _push_message(self, msg: str):
        self._messages.append(msg)
        if len(self._messages) > 20:
            self._messages = self._messages[-20:]

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"

    def quit(self):
        pygame.quit()
