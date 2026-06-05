# -*- coding: utf-8 -*-
"""
手势调试工具 — 运行此脚本查看实时捏合比率和状态
用法：venv/Scripts/python debug_gesture.py
操作：对着摄像头做捏合动作，观察控制台输出；按 Q 退出
"""
import math
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

PINCH_THRESH   = 0.09   # 当前触发阈值
RELEASE_THRESH = 0.14
SMOOTH_WINDOW  = 5
MIN_HOLD       = 3

hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.6,
)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

buf = deque(maxlen=SMOOTH_WINDOW)
hold_frames = 0
prev_grabbing = False
frame_n = 0

print("=" * 55)
print("调试模式：慢慢做捏合动作，观察 ratio 变化")
print(f"  捏合阈值: ratio < {PINCH_THRESH}  → GRABBING")
print(f"  松开阈值: ratio > {RELEASE_THRESH} → RELEASING")
print("=" * 55)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    state = "IDLE"
    ratio_raw = None
    ratio_smooth = None

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        idx = lm[8]; thm = lm[4]; wrist = lm[0]; mid = lm[9]

        pinch_d = math.hypot(idx.x - thm.x, idx.y - thm.y)
        palm_d  = math.hypot(wrist.x - mid.x, wrist.y - mid.y)
        ratio_raw = pinch_d / (palm_d + 1e-6)
        buf.append(ratio_raw)
        ratio_smooth = float(np.median(buf))

        is_pinch = ratio_smooth < PINCH_THRESH
        is_open  = ratio_smooth > RELEASE_THRESH

        if is_pinch:
            hold_frames += 1
        else:
            hold_frames = 0

        if prev_grabbing and is_open:
            state = "RELEASING"
            prev_grabbing = False
        elif hold_frames >= MIN_HOLD:
            state = "GRABBING"
            prev_grabbing = True
        else:
            state = "IDLE"

        # 画到摄像头预览
        h, w = frame.shape[:2]
        cx = int((idx.x + thm.x) / 2 * w)
        cy = int((idx.y + thm.y) / 2 * h)
        color = (0,255,0) if state=="GRABBING" else (0,255,255) if state=="RELEASING" else (180,180,180)
        cv2.circle(frame, (cx, cy), 14, color, 3)
        cv2.putText(frame, f"raw={ratio_raw:.3f}  smooth={ratio_smooth:.3f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, f"state={state}  hold={hold_frames}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"idx_x={idx.x:.2f}  idx_y={idx.y:.2f}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        mp.solutions.drawing_utils.draw_landmarks(
            frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)

        # 每5帧打印一次（避免刷屏）
        if frame_n % 5 == 0:
            bar = int((1 - min(ratio_smooth, 0.3) / 0.3) * 30)
            print(f"ratio={ratio_smooth:.3f} (raw={ratio_raw:.3f})  "
                  f"[{'#'*bar}{' '*(30-bar)}]  {state}  hold={hold_frames}")
    else:
        if frame_n % 30 == 0:
            print("--- 未检测到手部 ---")

    cv2.imshow("Gesture Debug (Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_n += 1

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n调试结束")
print(f"建议：如果捏合时 ratio 最小值 > {PINCH_THRESH}，需要降低 GESTURE_PINCH_THRESHOLD")
print(f"      如果 ratio 始终 < {RELEASE_THRESH}，需要提高 GESTURE_RELEASE_THRESHOLD")
