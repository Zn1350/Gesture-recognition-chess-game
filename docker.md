# 基于手势识别的中国象棋康复游戏 - Docker部署指南

## 项目简介
本项目是一个基于 Pygame 的手部康复象棋系统，支持摄像头手势识别和 AI 对弈。

## 快速开始

### 方法一：使用 Docker Compose（推荐）

```bash
# 进入项目目录
cd 终版

# 构建并启动容器
docker-compose up --build
```

### 方法二：使用 Docker 命令

```bash
# 进入项目目录
cd 终版

# 构建镜像
docker build -t rehab-chess .

# 运行容器（Linux/Mac）
docker run -it --rm \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  rehab-chess

# 运行容器（Windows - 使用 WSL2）
docker run -it --rm \
  --privileged \
  -v /dev/video0:/dev/video0 \
  rehab-chess
```

## 停止服务

```bash
# 停止并删除容器
docker-compose down

# 或使用 Docker 命令
docker stop rehab-chess-app
docker rm rehab-chess-app
```

## 技术栈

- **框架**: Pygame 2.6.1
- **语言**: Python 3.10
- **机器学习**: PyTorch 2.3.1, MediaPipe 0.10.14
- **图像处理**: OpenCV 4.11.0
- **界面**: Pygame

## 项目结构

```
终版/
├── main.py              # 主程序入口（桌面应用）
├── web/                 # Flask Web 应用（可选）
│   ├── app.py           # Web 应用入口
│   └── templates/       # HTML 模板
├── ai_engine.py         # AI 引擎
├── chess_game.py        # 象棋游戏逻辑
├── assessment.py        # 康复评估模块
├── gesture_rl.py        # 手势强化学习
├── vision.py            # 视觉感知模块
├── ui.py                # 图形界面模块
├── config.py            # 配置文件
├── model_improved.pth   # 神经网络模型
├── micro_dual.pth       # 轻量双头网络模型
├── requirements.txt     # 依赖列表
├── Dockerfile           # Docker 镜像配置
└── docker-compose.yml   # Docker Compose 配置
```

## 注意事项

1. **摄像头访问**: 需要在宿主机上允许容器访问摄像头设备
2. **图形界面**: Linux/Mac 需要共享 X11 显示；Windows 使用 WSL2
3. **模型文件**: 确保 `model_improved.pth` 和 `micro_dual.pth` 存在于项目根目录
4. **首次构建**: 首次构建镜像需要下载依赖，可能需要几分钟时间

## 启动验证

启动后可以通过以下方式验证服务是否正常运行：

```bash
# 查看容器日志
docker-compose logs

# 或使用 Docker 命令
docker logs rehab-chess-app
```

看到类似以下输出表示启动成功：
```
pygame 2.6.1 (SDL 2.28.4, Python 3.10.12)
Hello from the pygame community. https://www.pygame.org/contribute.html
```

## 快捷键

| 按键 | 功能 |
|------|------|
| R | 重新开始游戏 |
| Z | 悔棋 |
| ESC / Q | 退出游戏 |

```
# 构建镜像
docker build -t rehab-chess .

# 运行容器
docker-compose up
```

**注意**: 由于这是桌面应用，需要摄像头和图形界面支持。如果您的环境是 Linux 或 Mac，可以直接运行；如果是 Windows，建议使用 WSL2 或者使用 `web/app.py` 的 Web 版本。