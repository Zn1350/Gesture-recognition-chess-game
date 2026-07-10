# -*- coding: utf-8 -*-
"""
全局配置模块 — 集中管理所有可调参数，适配不同康复程度用户
"""

# ─── 摄像头配置 ───────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30

# ─── 窗口布局 ─────────────────────────────────────────────────────────────────
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 860
WINDOW_TITLE  = "手部康复象棋系统 | Hand Rehab Chinese Chess"

# 摄像头预览区（左上）
CAMERA_DISPLAY_RECT = (0, 0, 638, 637)

# 象棋对弈区（右上）
BOARD_DISPLAY_RECT = (643, 0, 637, 637)
BOARD_PADDING = 55                    # 棋盘左右内部空白，保证棋子距边缘至少 55px
BOARD_VERTICAL_PADDING_TOP = 55       # 棋盘上方内部空白
BOARD_VERTICAL_PADDING_BOTTOM = 55    # 棋盘下方内部空白
BOARD_GESTURE_MARGIN = 0.08           # 对应默认映射空白比例（兼容）

# 康复数据状态区（下方整行）
HUD_RECT = (0, 637, 1280, 223)

# ─── 手势识别灵敏度（核心可调参数）───────────────────────────────────────────
# 捏合距离阈值（相对于手掌宽度的比例，越小越灵敏）
# 范围建议：0.06（高灵敏/轻度康复）~ 0.15（低灵敏/重度康复）
GESTURE_PINCH_THRESHOLD  = 0.20   # 捏合判定阈值（抓取触发）
GESTURE_RELEASE_THRESHOLD = 0.30  # 松开判定阈值（释放触发），需大于捏合阈值以防抖

# 手势状态平滑窗口大小（帧数），越大越稳定但延迟越高
GESTURE_SMOOTH_WINDOW = 5

# 映射源：True 使用掌心（默认），False 使用食指指尖
GESTURE_USE_PALM = True

# 前端/界面显示坐标指数平滑系数（0~1），越大越平滑但延迟越高
GESTURE_DISPLAY_ALPHA = 0.45

# 手部检测置信度阈值
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE  = 0.5

# 最小捏合保持帧数（防止误触发）
GESTURE_MIN_HOLD_FRAMES = 2

# ─── 象棋棋盘参数 ─────────────────────────────────────────────────────────────
BOARD_COLS = 9   # 列
BOARD_ROWS = 10  # 行
BOARD_BG_COLOR   = (210, 180, 140)  # 棋盘背景色
BOARD_LINE_COLOR = (101,  67,  33)  # 棋盘线条色
PIECE_RADIUS = 22

# 棋子颜色
RED_COLOR    = (180,  30,  30)
BLACK_COLOR  = (30,   30,  30)
SELECT_COLOR = (0,   200,   0)   # 选中高亮
LEGAL_COLOR  = (0,   150, 255)   # 可落子高亮

# ─── AI 难度参数 ──────────────────────────────────────────────────────────────
# 技能评级 1-5 对应 AI 搜索深度
AI_DEPTH_MAP = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}

# 各评级下 AI 随机失误概率（让棋幅度）
# 技能越低 → AI 出错概率越高 → 玩家更容易赢
AI_BLUNDER_RATE = {1: 0.50, 2: 0.35, 3: 0.20, 4: 0.08, 5: 0.00}

# 是否启用训练好的神经网络评估器
USE_NN_EVALUATOR = True
NN_EVAL_MODEL_PATH = "model_improved.pth"

# AI 思考延迟（秒）
AI_THINK_DELAY = 0.8

# ─── 康复评估参数 ─────────────────────────────────────────────────────────────
# 技能评级阈值（error_rate, avg_move_time_s）
# 低失误率 + 快速操作 → 高评级
SKILL_RATING_THRESHOLDS = [
    # (max_error_rate, max_avg_time) → rating
    (0.05, 8.0,  5),
    (0.10, 12.0, 4),
    (0.20, 18.0, 3),
    (0.35, 25.0, 2),
    (1.00, 999,  1),
]

# ─── 颜色常量 ─────────────────────────────────────────────────────────────────
COLOR_BG          = (30,  30,  40)
COLOR_WHITE       = (255, 255, 255)
COLOR_GRAY        = (160, 160, 160)
COLOR_DARK_GRAY   = (60,  60,  70)
COLOR_GREEN       = (80,  200, 120)
COLOR_YELLOW      = (255, 220,  50)
COLOR_RED         = (220,  60,  60)
COLOR_CYAN        = (50,  200, 200)
COLOR_ORANGE      = (255, 160,  50)

# ─── 字体路径（支持中文）────────────────────────────────────────────────────
# 优先使用系统中文字体；若找不到则由 ui.py 回退到 pygame 内置字体
FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",     # 黑体
    "C:/Windows/Fonts/simsun.ttc",     # 宋体
]
FONT_SIZE_LARGE  = 28
FONT_SIZE_MEDIUM = 20
FONT_SIZE_SMALL  = 15
FONT_SIZE_PIECE  = 20  # 棋子字符大小
