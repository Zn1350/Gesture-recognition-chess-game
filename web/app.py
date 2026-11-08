"""Local Flask interface for one Gomoku game."""

import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion import ExpressionRecognizer
from gomoku_session import GomokuSession


def create_app(session=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=128 * 1024, TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])
    game = session if session is not None else GomokuSession()
    lock = threading.Lock()
    inference_lock = threading.Lock()
    recognizer = None

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/state")
    def state():
        with lock:
            game.tick()
            return jsonify(game.state())

    @app.post("/api/move")
    def move():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or any(type(data.get(key)) is not int for key in ("row", "col")):
            return jsonify(error="行和列必须是整数"), 400
        with lock:
            game.tick()
            if not game.move(data["row"], data["col"]):
                return jsonify(error="该位置当前不可落子"), 409
            return jsonify(game.state())

    @app.post("/api/restart")
    def restart():
        with lock:
            game.restart()
            return jsonify(game.state())

    @app.post("/api/undo")
    def undo():
        with lock:
            game.tick()
            if not game.undo():
                return jsonify(error="当前没有可撤回的落子"), 409
            return jsonify(game.state())

    @app.post("/api/camera")
    def camera():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or type(data.get("enabled")) is not bool:
            return jsonify(error="需要表情辅助开关状态"), 400
        with lock:
            game.set_camera(data["enabled"])
            return jsonify(game.state())

    @app.post("/api/expression")
    def expression():
        nonlocal recognizer
        token = request.headers.get("X-Observation-Token")
        captured_at = time.monotonic()
        with lock:
            if not token or token != game.sample_token:
                return jsonify(message="当前回合不接收此帧"), 409
        if request.mimetype != "image/jpeg":
            return jsonify(error="需要 JPEG 图像"), 415
        if not inference_lock.acquire(blocking=False):
            return jsonify(message="正在处理上一帧")
        try:
            import cv2
            import numpy as np

            jpeg = request.get_data()
            if not jpeg.startswith(b"\xff\xd8"):
                return jsonify(error="无法读取 JPEG 图像"), 400
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or max(frame.shape[:2]) > 800:
                return jsonify(error="无法读取图像或图像尺寸过大"), 400
            if recognizer is None:
                recognizer = ExpressionRecognizer()
            observation = recognizer.recognize(frame)
        finally:
            inference_lock.release()
        with lock:
            if not game.observe(token, observation, captured_at):
                return jsonify(message="表情样本已过期")
        if not recognizer.available:
            return jsonify(message="表情模型不可用，请检查模型文件")
        return jsonify(label=observation.label if observation else None)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
