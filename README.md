# 基于手势识别的中国象棋康复游戏

基于计算机视觉与强化学习的手部康复训练系统，通过手势控制中国象棋游戏，帮助中风患者或手部损伤患者进行康复训练。

## 项目概述

本系统融合五大核心模块，为手部康复训练提供智能化解决方案：

| 模块 | 功能描述 |
|------|----------|
| **视觉感知** | 基于 MediaPipe 的实时手势识别，支持捏合抓取、松开释放、剪刀手等手势 |
| **象棋逻辑** | 完整的中国象棋规则实现，支持走棋、吃子、悔棋等操作 |
| **AI博弈** | 基于神经网络评估器的智能对手，可根据患者技能水平动态调整难度 |
| **康复评估** | 实时追踪患者表现，计算技能评级和康复进度 |
| **可视化界面** | 直观的游戏界面，实时显示摄像头画面、棋盘状态和康复数据 |

## 技术栈

- **Python 3.10+** - 主开发语言
- **Pygame** - 游戏界面渲染
- **OpenCV** - 摄像头视频处理
- **MediaPipe** - 手部关键点检测与手势识别
- **PyTorch** - 神经网络评估器
- **Flask** - Web 服务
- **强化学习** - 动态手势阈值自适应（Q-Learning）
- **深度学习** - 棋局评估神经网络（价值头+策略头）
- **预训练模型** - `model_improved.pth`（棋局评估）

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- 支持 USB 摄像头（推荐分辨率 640x480）
- Windows 10/11（其他系统需调整字体路径）

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-repo/gesture-rehab-chess.git
   cd gesture-rehab-chess
   ```

2. **创建虚拟环境**（推荐）
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # 或
   source venv/bin/activate  # Linux/Mac
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   # 如有GPU，则使用pip install -r requirements_gpu.txt
   ```

4. **运行主程序**
   ```bash
   python main.py
   ```

## 游戏操作指南

### 手势控制

| 手势 | 操作说明 |
|------|----------|
| **捏合抓取** | 在棋盘上握紧拳头选中己方棋子 |
| **松开释放** | 将选中的棋子放置到目标位置 |
| **剪刀手** | 保持3秒触发悔棋功能 |

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `R` | 重新开始新对局 |
| `Z` | 悔棋（需玩家回合） |
| `ESC` / `Q` | 退出游戏 |

### 游戏规则

1. 玩家执红棋先行，AI执黑棋后手
2. 捏合手势选中己方棋子，松开手势落子
3. 只能移动到高亮显示的合法位置
4. 完成有效走棋后可使用剪刀手悔棋

## 配置说明

主要配置项位于 `config.py`：

```python
# 手势灵敏度调整（根据康复程度调整）
GESTURE_PINCH_THRESHOLD = 0.20   # 捏合判定阈值，越小越灵敏
GESTURE_RELEASE_THRESHOLD = 0.30 # 松开判定阈值

# AI 难度映射
AI_DEPTH_MAP = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}  # 技能等级→搜索深度

# 康复评估阈值（失误率, 平均时间）
SKILL_RATING_THRESHOLDS = [
    (0.05, 8.0, 5),   # 低失误+快操作 → 5级（最高）
    (0.10, 12.0, 4),
    (0.20, 18.0, 3),
    (0.35, 25.0, 2),
    (1.00, 999, 1),    # 高失误+慢操作 → 1级（最低）
]
```

## 项目结构

```
├── main.py             # 主程序入口
├── config.py           # 全局配置
├── ui.py               # Pygame 界面渲染
├── chess_game.py       # 象棋游戏逻辑
├── vision.py           # 视觉感知模块
├── ai_engine.py        # AI 博弈引擎
├── assessment.py       # 康复评估模块
├── gesture_rl.py       # 手势强化学习控制器
├── nn_evaluator.py     # 神经网络评估器
├── features.py         # 棋局特征提取
├── model_improved.pth  # 预训练神经网络模型
├── web/                # Web 服务模块
│   ├── app.py          # Flask 应用
│   └── templates/
│       └── index.html  # 前端页面
├── notebooks/          # Jupyter 笔记本
│   └── train_on_kaggle.ipynb
└── requirements.txt    # 依赖列表
└── requirements_gpu.txt    # 依赖列表
```

## 核心功能

### 1. 自适应手势识别

系统采用强化学习算法，根据用户操作反馈动态调整手势识别阈值：
- 操作成功 → 提高阈值要求（更严格）
- 操作失误 → 降低阈值（更宽松）
- 停留超时 → 自动降低阈值

### 2. 动态难度 AI

AI 会根据患者实时技能评分自动调整：
- **搜索深度**：技能越高，搜索越深
- **失误概率**：技能越低，AI故意失误概率越高
- **神经网络评估**：使用预训练模型评估棋局优劣

### 3. 康复评估指标

系统实时追踪以下康复指标：
- **技能评分**：综合评估操作准确性和速度
- **移动次数**：累计有效走棋数
- **失误率**：非法操作占比
- **平均用时**：每步操作耗时



## AI技术应用

### 1. 神经网络棋局评估器

本系统采用**深度神经网络**对棋局状态进行评估，核心模型文件说明：

| 模型文件 | 功能描述 | 技术特点 |
|----------|----------|----------|
| `model_improved.pth` | 主棋局评估器 | 多层卷积神经网络，输出棋局优劣评分 |


**技术实现要点：**
- 输入：8x8特征平面（棋子类型、位置、威胁关系等）
- 输出：[-1, 1]区间评分（正为红方优势，负为黑方优势）
- 训练方式：自我对弈强化学习 + 监督学习微调

### 2. 动态难度AI引擎

AI博弈系统具备**自适应难度调整**能力：

```python
# 核心机制
def get_move(game_state, skill_score):
    # 根据实时技能评分动态调整
    depth = AI_DEPTH_MAP[skill_level]        # 搜索深度自适应
    blunder_rate = AI_BLUNDER_RATE[skill_level]  # 失误概率自适应
    
    # 结合神经网络评估
    if USE_NN_EVALUATOR:
        eval_score = self.nn_evaluator.evaluate(game_state)
    else:
        eval_score = self.handcrafted_evaluate(game_state)
    
    # Minimax搜索 + α-β剪枝
    best_move = minimax(game_state, depth, eval_score)
    
    # 概率性失误（让棋机制）
    if random.random() < blunder_rate:
        best_move = self._select_blunder_move()
    
    return best_move
```

**关键技术：**
- **Minimax算法**：带深度限制的极大极小搜索
- **α-β剪枝**：优化搜索效率，减少无效分支
- **动态放水机制**：根据患者技能水平智能让棋

### 3. 强化学习手势阈值自适应

系统采用**Q-Learning算法**实时优化手势识别参数：

**状态空间：**
- 当前捏合阈值、松开阈值、保持帧数

**动作空间：**
- 微调阈值（±0.01）、调整保持帧数（±1）

**奖励机制：**
| 事件 | 奖励值 | 说明 |
|------|--------|------|
| 抓取成功 | +1.0 | 有效选中己方棋子 |
| 释放成功 | +1.0 | 有效完成落子 |
| 抓取失败 | -0.5 | 误触（无己方棋子位置） |
| 释放失败 | -0.5 | 松开位置无效 |
| 停留超时 | -0.3 | 触发自适应下调阈值 |

**学习流程：**
```
观测手势交互 → 记录成功/失败 → 更新Q值 → 动态调整阈值 → 优化识别效果
```

### 4. 技能评级算法

康复评估模块采用**多维度加权评分**：

```python
def calculate_skill_score(stats):
    # 权重配置
    weights = {
        'error_rate': 0.4,    # 失误率权重
        'avg_time': 0.3,      # 平均用时权重
        'move_count': 0.2,    # 有效移动次数权重
        'captures': 0.1,      # 吃子数权重
    }
    
    # 归一化计算
    score = (1 - stats.error_rate) * weights['error_rate'] + \
            max(0, 1 - stats.avg_time / 30) * weights['avg_time'] + \
            min(1, stats.move_count / 50) * weights['move_count'] + \
            min(1, stats.captures / 10) * weights['captures']
    
    return int(score * 100)



## 开发说明

### 调试工具

```bash
# 调试手势识别（CLI模式）
python debug_gesture_cli.py

# 调试手势识别（可视化模式）
python debug_gesture.py
```

### 训练神经网络

```bash
# 生成训练数据
python gen_data.py

# 训练微型双头网络
python train_micro.py
```

## 注意事项

1. 确保摄像头正常工作且无遮挡
2. 建议在光线充足的环境下使用
3. 首次运行可能需要加载模型，耗时稍长
4. 若摄像头无法打开，程序会自动切换到无摄像头模式

## 技术亮点总结

| 技术维度 | 实现方案 | 创新点 |
|----------|----------|--------|
| **机器学习** | Q-Learning手势阈值自适应 | 实时优化识别参数，适配不同用户 |
| **深度学习** | 卷积神经网络棋局评估 | 端到端学习棋局价值，超越手工规则 |
| **博弈论** | 动态难度Minimax搜索 | 根据患者水平智能调整，兼顾康复与体验 |
| **系统集成** | 多模块协同工作 | 视觉→手势→棋局→评估全链路闭环 |
| **可扩展性** | 模块化设计 | 支持更换评估模型、调整规则参数 |

## License

MIT License