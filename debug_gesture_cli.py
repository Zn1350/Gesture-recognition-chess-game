# -*- coding: utf-8 -*-
"""
无界面手势调试工具 — 纯控制台输出 ratio 值
用法：venv/Scripts/python debug_gesture_cli.py
操作：对着摄像头做捏合动作，看 ratio 数值，10秒后自动退出
"""
import math, time
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

PINCH_THRESH   = 0.09
RELEASE_THRESH = 0.14
SMOOTH_WINDOW  = 5
MIN_HOLD       = 3

hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.6,
)
cap = cv2.VideoCapture(0)

buf = deque(maxlen=SMOOTH_WINDOW)
hold_frames = 0
prev_grabbing = False
frame_n = 0
start = time.time()
RUN_SECS = 15

print("=" * 60)
print(f"运行 {RUN_SECS} 秒，请对摄像头做捏合/张开动作")
print(f"  捏合阈值: ratio < {PINCH_THRESH}  → GRABBING")
print(f"  松开阈值: ratio > {RELEASE_THRESH} → RELEASING")
print("=" * 60)

while time.time() - start < RUN_SECS:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    state = "IDLE"
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        idx = lm[8]; thm = lm[4]; wrist = lm[0]; mid = lm[9]
        pinch_d = math.hypot(idx.x - thm.x, idx.y - thm.y)
        palm_d  = math.hypot(wrist.x - mid.x, wrist.y - mid.y)
        ratio_raw = pinch_d / (palm_d + 1e-6)
        buf.append(ratio_raw)
        smooth = float(np.median(buf))

        is_pinch = smooth < PINCH_THRESH
        is_open  = smooth > RELEASE_THRESH
        hold_frames = hold_frames + 1 if is_pinch else 0

        if prev_grabbing and is_open:
            state = "RELEASING"; prev_grabbing = False
        elif hold_frames >= MIN_HOLD:
            state = "GRABBING"; prev_grabbing = True

        if frame_n % 3 == 0:
            bar = int((1 - min(smooth, 0.3) / 0.3) * 25)
            flag = " *** GRAB ***" if state == "GRABBING" else (" ** RELEASE **" if state == "RELEASING" else "")
            print(f"  ratio={smooth:.3f} raw={ratio_raw:.3f}  "
                  f"[{'#'*bar}{' '*(25-bar)}]  {state}{flag}  hold={hold_frames}")
    else:
        if frame_n % 20 == 0:
            print("  (未检测到手部)")
    frame_n += 1

cap.release()
hands.close()
print("=" * 60)
print("调试完成。根据上面的 ratio 值决定阈值：")
print(f"  你的捏合最小 ratio 应低于 GESTURE_PINCH_THRESHOLD({PINCH_THRESH})")
print(f"  你的张开最大 ratio 应高于 GESTURE_RELEASE_THRESHOLD({RELEASE_THRESH})")
