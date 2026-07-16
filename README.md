# 手势控制中国象棋康复训练系统

这是一个基于计算机视觉、手势识别和人工智能的中国象棋康复训练项目。它同时提供了桌面版 Pygame 程序和网页版 Flask 应用，用户可以通过手势控制棋子，系统会根据操作表现实时评估康复状态，并通过自适应 AI 和手势 RL 阈值调整提升训练体验。

## 项目定位

该项目的目标是把中国象棋与手部康复训练结合起来：

- 通过摄像头识别手势完成棋子抓取、落子和悔棋操作
- 通过实时统计和 Q-Learning 评估用户的操作表现
- 通过动态放水 AI 让对局难度与用户水平更匹配
- 既可作为桌面交互程序使用，也可部署为浏览器版本进行演示

## 当前实现内容

### 1. 桌面版程序

入口文件：`main.py`

特点：
- 使用 Pygame 渲染界面
- 使用 OpenCV + MediaPipe 实现手势识别
- 支持红方玩家通过手势走棋，黑方由 AI 下棋
- 支持键盘快捷键：`R` 重新开始、`Z` 悔棋、`ESC/Q` 退出
- 支持“剪刀手保持 3 秒”触发悔棋

### 2. Web 版程序

入口文件：`web/app.py`

特点：
- 使用 Flask 提供后端服务
- 前端通过浏览器中的 MediaPipe JS 完成手势识别
- 后端负责棋局逻辑、康复评分、AI 走棋与 RL 阈值更新
- 访问地址通常为：`http://localhost:5000`

## 技术栈

- Python 3.10+（推荐）
- Pygame：桌面界面
- OpenCV：图像读取与显示
- MediaPipe：手部关键点与手势识别
- PyTorch：神经网络评估器支持
- Flask：Web 服务
- NumPy：数值计算与特征处理

## 依赖安装

在项目根目录执行：

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

如果需要使用 GPU 相关依赖，也可以参考：

```bash
pip install -r requirements_gpu.txt
```

## 运行方式

### 运行桌面版

```bash
python main.py
```

启动后会打开摄像头窗口和棋盘界面，用户可通过手势完成对局。

### 运行 Web 版

```bash
python web/app.py
```

然后在浏览器中打开：

```text
http://localhost:5000
```

## 操作说明

### 手势控制

- 捏合并保持一段时间：选中己方棋子
- 松开：将选中的棋子落到目标位置
- 仅当落点是合法走法时，棋子才会真正移动
- 完成一次有效走棋后，保持剪刀手 3 秒可触发悔棋

### 键盘控制

- `R`：重新开始新对局
- `Z`：悔棋
- `ESC` / `Q`：退出程序

## 项目结构

当前仓库包含桌面版交互程序、网页版演示服务、训练脚本以及模型与数据资源，整体结构可分为以下几部分：

```text
.
├── main.py                  # 桌面版主程序入口
├── config.py                # 全局配置与参数定义
├── ui.py                    # Pygame 界面渲染与棋盘显示
├── chess_game.py            # 中国象棋规则与棋局逻辑
├── vision.py                # MediaPipe 手势识别与状态机
├── ai_engine.py             # 自适应 AI 引擎
├── assessment.py            # 康复评估与棋力评分逻辑
├── gesture_rl.py            # 手势阈值自适应 RL 模块
├── features.py              # 棋局特征提取与状态编码
├── nn_evaluator.py          # 神经网络评估器相关代码
├── gen_data.py              # 训练数据生成脚本
├── train_micro.py           # 微型网络训练脚本
├── model_improved.pth       # 预训练模型文件
├── micro_dual.pth           # 微型双头网络模型文件
├── selfplay_data/           # 自对弈数据集目录
│   ├── boards.npy
│   ├── features.npy
│   ├── move_scores.npy
│   ├── moves_order.npy
│   └── z_values.npy
├── micro_dual/              # PyTorch 模型资源目录
├── notebooks/               # 训练与实验 notebook
├── web/                     # Flask 网页版实现
│   ├── app.py               # Web 后端入口
│   └── templates/
│       └── index.html       # Web 前端页面
├── requirements.txt         # Python 依赖列表
├── requirements_gpu.txt     # GPU 环境依赖列表
├── requiments安装步骤.txt   # 本地安装说明
└── README.md                # 项目说明文档
```

## 核心机制

### 1. 手势识别与状态机

项目使用 MediaPipe 检测手部关键点，通过捏合比例判断手势状态：

- `IDLE`：未触发手势
- `GRABBING`：正在捏合，表示选子
- `RELEASING`：刚松开，表示落子

此外，系统还会识别剪刀手，用于悔棋操作。

### 2. 自适应 AI

AI 会根据实时棋力评分调整走法策略：

- 棋力分越低，AI 的放水概率越高，故意失误更明显
- 棋力分越高，AI 越接近正常搜索和更强的落子判断
- 同时会结合神经网络评估器和手工评估进行走法打分

### 3. 康复评估

系统会实时记录：

- 总走棋数
- 非法操作次数
- 吃子数
- 平均每步耗时
- 手势准确率

基于这些指标，程序会计算一个棋力分和对应的技能标签，例如：

- 初学康复
- 进阶练习
- 稳定发展
- 熟练掌握
- 康复达人

### 4. 手势 RL 自适应

手势模块中嵌入了一个轻量级 Q-Learning 机制，用于根据用户操作反馈动态调整捏合阈值、松开阈值和保持帧数：

- 连续成功会让系统更趋向于收紧阈值
- 失误率较高时会放宽阈值，降低操作难度
- 停留超时也会触发阈值下调，避免用户长时间卡住

## 训练与数据

仓库中包含训练相关脚本：

```bash
python gen_data.py
python train_micro.py
```

这些脚本可用于生成自我对弈数据或训练轻量双头网络模型。

## 使用注意

1. 建议在光线较好、摄像头无遮挡的环境下运行。
2. 首次运行可能需要加载模型文件，耗时较长。
3. 若摄像头无法打开，程序会显示无摄像头提示，但界面仍可进入基本流程。
4. 由于本项目依赖手势识别与摄像头输入，实际体验会受到环境光照和手部姿态影响。

## 许可证

当前仓库未提供单独的许可证文件，若需要正式发布请先确认授权说明。