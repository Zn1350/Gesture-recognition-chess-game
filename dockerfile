# 手部康复象棋系统 - Docker 镜像配置
# 基于 Python 3.10，支持桌面应用（main.py）

FROM python:3.10-slim

# 维护者信息
LABEL maintainer="Student <student@example.com>"
LABEL description="手部康复象棋系统 - Hand Rehab Chinese Chess"

# 设置工作目录
WORKDIR /app

# 安装系统依赖（OpenCV、MediaPipe、Pygame 需要的底层库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1\
    libgomp1 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用国内镜像加速）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目所有文件
COPY . .

# 验证模型文件存在（构建时检查）
RUN ls -la *.pth || echo "Note: Model files may need to be placed manually"

# 设置环境变量
ENV PYTHONPATH=/app
ENV DISPLAY=:0

# 创建非特权用户（可选，但推荐）
RUN useradd -m chessuser
USER chessuser

# 启动命令（桌面应用入口）
CMD ["python", "main.py"]