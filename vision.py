# -*- coding: utf-8 -*-
"""
视觉感知模块
基于 MediaPipe 21点手部关键点，计算食指-拇指捏合比率
实现 IDLE / GRABBING / RELEASING 三态手势状态机
"""

from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config


class GestureState(Enum):
    IDLE      = auto()   # 手未捏合
    GRABBING  = auto()   # 捏合保持中（抓取/选子/拖动）
    RELEASING = auto()   # 刚刚松开（落子触发，仅维持1帧）


@dataclass
class GestureResult:
    state: GestureState = GestureState.IDLE
    # 食指指尖在摄像头画面中的归一化坐标 (0~1)
    norm_x: float = 0.0
    norm_y: float = 0.0
    # 摄像头像素坐标
    pixel_x: int = 0
    pixel_y: int = 0
    # 捏合比率（越小越捏合）
    pinch_ratio: float = 1.0
    # 手部检测到
    hand_detected: bool = False
    # 剪刀手（食指+中指伸直，其余弯曲）
    is_scissors: bool = False


class VisionModule:
    """
    摄像头读取 + MediaPipe 手势识别
    线程安全：设计为单线程使用，由主循环每帧调用 update()
    """

    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
        )
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        # 捏合比率平滑队列
        self._ratio_buf: deque = deque(maxlen=config.GESTURE_SMOOTH_WINDOW)
        # 位置平滑队列（减少手势映射抖动）
        self._pos_buf_x: deque = deque(maxlen=config.GESTURE_SMOOTH_WINDOW)
        self._pos_buf_y: deque = deque(maxlen=config.GESTURE_SMOOTH_WINDOW)
        # 是否使用掌心作为映射基准（可运行时切换）
        self._use_palm = getattr(config, 'GESTURE_USE_PALM', True)

        # 状态机
        self._state = GestureState.IDLE
        self._grab_hold_frames = 0   # 连续捏合帧计数（防抖）
        self._prev_was_grabbing = False

        self._last_result = GestureResult()
        self._frame: Optional[np.ndarray] = None

        # 剪刀手防抖：连续帧计数
        self._scissors_frames = 0

    @staticmethod
    def _is_finger_extended(lm, tip: int, pip: int) -> bool:
        """指尖 y 坐标小于近端指间关节 y 坐标即视为伸直（屏幕坐标 y 向下）"""
        return lm[tip].y < lm[pip].y

    @staticmethod
    def _is_finger_bent(lm, tip: int, pip: int) -> bool:
        return lm[tip].y > lm[pip].y

    def _detect_scissors(self, lm) -> bool:
        """
        剪刀手判断：
        - 食指(8/7)、中指(12/11) 伸直
        - 无名指(16/15)、小拇指(20/19) 弯曲
        - 拇指不做限制（自然弯曲即可）
        """
        index_up  = self._is_finger_extended(lm, 8,  6)
        middle_up = self._is_finger_extended(lm, 12, 10)
        ring_down = self._is_finger_bent(lm, 16, 14)
        pinky_down= self._is_finger_bent(lm, 20, 18)
        return index_up and middle_up and ring_down and pinky_down

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def update(self) -> Tuple[GestureResult, Optional[np.ndarray]]:
        """
        读取一帧摄像头图像，运行手势识别
        返回 (GestureResult, annotated_bgr_frame)
        """
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return self._last_result, self._frame

        # 水平镜像：更直观（摄像头镜像）
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        annotated = frame.copy()
        result = GestureResult()

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            result.hand_detected = True

            # 绘制手部骨架
            mp.solutions.drawing_utils.draw_landmarks(
                annotated,
                landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

            h, w = frame.shape[:2]
            lm = landmarks.landmark

            # 食指指尖 (8) 和拇指指尖 (4)
            idx_tip  = lm[8]
            thm_tip  = lm[4]
            # 手腕 (0) 和中指根 (9) 构成手掌尺寸参考
            wrist    = lm[0]
            mid_mcp  = lm[9]

            # 归一化捏合距离 / 手掌参考长度
            pinch_d = math.hypot(idx_tip.x - thm_tip.x, idx_tip.y - thm_tip.y)
            palm_d  = math.hypot(wrist.x  - mid_mcp.x,  wrist.y  - mid_mcp.y)
            ratio   = pinch_d / (palm_d + 1e-6)

            self._ratio_buf.append(ratio)
            smooth_ratio = float(np.median(self._ratio_buf))

            # 选择映射基准（掌心或指尖），减少因指尖抖动导致的位置误差
            palm_x = (wrist.x + mid_mcp.x) / 2.0
            palm_y = (wrist.y + mid_mcp.y) / 2.0
            tip_x = idx_tip.x
            tip_y = idx_tip.y

            src_x = palm_x if self._use_palm else tip_x
            src_y = palm_y if self._use_palm else tip_y

            # 位置平滑：使用中位数滤波减少抖动
            self._pos_buf_x.append(src_x)
            self._pos_buf_y.append(src_y)
            smooth_x = float(np.median(self._pos_buf_x))
            smooth_y = float(np.median(self._pos_buf_y))

            result.norm_x = smooth_x
            result.norm_y = smooth_y
            result.pixel_x = int(smooth_x * w)
            result.pixel_y = int(smooth_y * h)
            result.pinch_ratio = smooth_ratio

            # ── 剪刀手检测（防抖：连续3帧才确认）──────────────────────────
            if self._detect_scissors(lm):
                self._scissors_frames += 1
            else:
                self._scissors_frames = 0
            result.is_scissors = self._scissors_frames >= 3

            # ── 状态机转换 ─────────────────────────────────────────────────
            is_pinching = smooth_ratio < config.GESTURE_PINCH_THRESHOLD
            is_open     = smooth_ratio > config.GESTURE_RELEASE_THRESHOLD

            if is_pinching:
                self._grab_hold_frames += 1
            else:
                self._grab_hold_frames = 0

            if self._prev_was_grabbing and is_open:
                # 刚松开 → 触发落子
                result.state = GestureState.RELEASING
                self._state  = GestureState.IDLE
                self._prev_was_grabbing = False
            elif self._grab_hold_frames >= config.GESTURE_MIN_HOLD_FRAMES:
                result.state = GestureState.GRABBING
                self._state  = GestureState.GRABBING
                self._prev_was_grabbing = True
            else:
                result.state = GestureState.IDLE
                self._state  = GestureState.IDLE

            # 绘制捏合指示圆
            cx = int((idx_tip.x + thm_tip.x) / 2 * w)
            cy = int((idx_tip.y + thm_tip.y) / 2 * h)
            color = (0, 255, 0) if result.state == GestureState.GRABBING else \
                    (0, 255, 255) if result.state == GestureState.RELEASING else \
                    (200, 200, 200)
            cv2.circle(annotated, (cx, cy), 12, color, 2)
            cv2.putText(annotated, f"{result.state.name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated, f"ratio:{smooth_ratio:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if result.is_scissors:
                cv2.putText(annotated, "SCISSORS ✌", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            # 未检测到手，重置状态
            self._state = GestureState.IDLE
            self._prev_was_grabbing = False
            self._grab_hold_frames = 0
            self._scissors_frames = 0

        self._last_result = result
        self._frame = annotated
        return result, annotated

    def release(self):
        self._cap.release()
        self._hands.close()

    def set_thresholds(
        self,
        pinch_threshold:   float,
        release_threshold: float,
        hold_frames:       int,
    ):
        """由 GestureRL 实时调用，动态更新捏合判定阈值"""
        config.GESTURE_PINCH_THRESHOLD   = pinch_threshold
        config.GESTURE_RELEASE_THRESHOLD = release_threshold
        config.GESTURE_MIN_HOLD_FRAMES   = hold_frames

    def set_use_palm(self, use_palm: bool):
        """在运行时切换映射源（掌心或指尖）。"""
        self._use_palm = bool(use_palm)

    @property
    def is_camera_open(self) -> bool:
        return self._cap.isOpened()
