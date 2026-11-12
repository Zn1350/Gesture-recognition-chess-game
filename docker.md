# 五子棋表情辅助对弈 · Docker 使用说明

容器运行 `main.py` 桌面版。玩家执黑，点击 15×15 棋盘落子。摄像头仅用于可选的表情辅助。

## 在 Linux 上启动

在能够访问宿主机 X11 显示和 `/dev/video0` 摄像头的环境中，从项目根目录运行以下命令。

```bash
docker compose -f docker_compose.yml up --build
```

只需要鼠标对弈时，可单独启动桌面容器，省去摄像头设备映射。

```bash
docker build -f dockerfile -t rehab-gomoku .
docker run --rm -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix rehab-gomoku
```

用 Compose 启动后，可用 `docker compose -f docker_compose.yml down` 停止服务。

Windows 上可以直接启动桌面版或本地网页版。 [项目运行说明](README.md)

## 操作

- 点击棋盘落子，按 `R` 开新局，按 `Z` 撤回最近一轮，按 `Esc` 退出。
- 有摄像头时，按 `C` 开关表情辅助。YuNet 和 MobileFaceNet 模型已随镜像打包。
