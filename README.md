# 五子棋表情辅助对弈

玩家执黑先行，AI 执白。棋盘为 15×15 路，横、竖或斜向连成至少五枚同色棋子即获胜，棋盘填满且无人获胜时为和棋。

## 运行

在项目根目录使用 Python 3.10–3.12 安装依赖。

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

桌面版点击棋盘落子。按 `C` 开关表情辅助，`R` 开新局，`Z` 撤回最近一轮，`Esc` 退出。其他平台可用 `python3 -m venv .venv`，将后续 Python 路径改为 `.venv/bin/python`。

本地网页版用 `.venv\Scripts\python web/app.py` 启动，再打开 <http://127.0.0.1:5000>。一个进程维护一局，摄像头由浏览器授权，抽样图像经本机回环交给 Python 进程处理。桌面和 Web 分别运行自己的棋局。

## 表情辅助

- 摄像头默认关闭。开启后，YuNet 检测单人脸，MobileFaceNet 输出七类表情的分类分数。有效样本仅来自当前玩家回合，未保存为图像文件。
- 在玩家合法落子后，系统用当前回合内通过筛选的样本判断是否将下一手 AI 的难度降低一档。负向分数综合生气、厌恶、害怕和难过四类。低置信度、短暂表情和缺失样本不会触发降难。该信号是表情分类结果，用于对弈辅助。
- 难度从标准开始，经适中降到轻松。档位调整有两个 AI 回合的冷却间隔。关闭辅助、悔棋或开启新局后回到标准难度。
- AI 按档位设置候选宽度、搜索深度和节点预算，并在各档共用一步获胜和阻止玩家下一手成五的检查。关闭摄像头或推理不可用时仍能用鼠标对弈。

模型已随项目提供，可以离线运行。模型来源和许可见 [models/README.md](models/README.md)。

## 代码与测试

`gomoku_game.py` 实现五子棋规则，`gomoku_ai.py` 实现对弈搜索，`emotion.py` 实现本机表情推理和降难策略，`gomoku_session.py` 负责两端共用的回合、采样和搜索状态。

```powershell
.venv\Scripts\python -m unittest discover -s tests
```

原中国象棋研究代码和训练数据保留在仓库中，新的对弈入口不加载其手势或棋力模型。
