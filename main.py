"""Pygame entry point for the emotion-adaptive Gomoku game."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pygame

from emotion import ExpressionRecognizer
from gomoku_game import BLACK, BOARD_SIZE
from gomoku_session import DECISIONS, DIFFICULTIES, GomokuSession


BOARD_ORIGIN = (42, 89)
CELL = 37
FACE_LABELS = {
    "angry": "生气", "disgust": "厌恶", "fearful": "害怕", "happy": "高兴",
    "sad": "难过", "surprised": "惊讶", "neutral": "中性",
}


def chinese_font_path():
    if os.name == "nt":
        folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
            path = folder / name
            if path.is_file():
                return str(path)
        return None
    return pygame.font.match_font(("Noto Sans CJK SC", "PingFang SC", "WenQuanYi Zen Hei"))


class Camera:
    """Read and classify in a worker so inference does not stall the game UI."""

    def __init__(self):
        self.capture = None
        self.recognizer = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gomoku-camera")

    def read(self, token):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
        ok, frame = self.capture.read()
        captured_at = time.monotonic()
        if not ok:
            return token, captured_at, None, None, "摄像头不可用"
        if self.recognizer is None:
            self.recognizer = ExpressionRecognizer()
        observation = self.recognizer.recognize(frame)
        preview = cv2.cvtColor(cv2.resize(cv2.flip(frame, 1), (280, 198)), cv2.COLOR_BGR2RGB)
        status = "等待有效面部图像"
        if not self.recognizer.available:
            status = "表情模型不可用"
        elif observation is not None:
            status = "表情线索 · " + FACE_LABELS.get(observation.label, observation.label)
        return token, captured_at, observation, preview, status

    def _release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def stop(self):
        self.executor.submit(self._release)

    def close(self):
        self.stop()
        self.executor.shutdown(wait=False)


class DesktopApp:
    def __init__(self):
        self.session = GomokuSession()
        self.camera = Camera()
        self.capture_job = None
        self.next_sample = 0.0
        self.preview = None
        self.camera_status = "表情辅助已关闭"
        self.screen = pygame.display.set_mode((936, 650))
        pygame.display.set_caption("五子棋 · 表情辅助")
        font_path = chinese_font_path()
        self.title = pygame.font.Font(font_path, 28)
        self.font = pygame.font.Font(font_path, 17)
        self.camera_button = pygame.Rect(629, 222, 280, 36)
        self.restart_button = pygame.Rect(629, 165, 134, 36)
        self.undo_button = pygame.Rect(775, 165, 134, 36)

    def toggle_camera(self):
        enabled = not self.session.camera_enabled
        self.session.set_camera(enabled)
        self.preview = None
        self.camera_status = "正在打开摄像头…" if enabled else "表情辅助已关闭"
        if not enabled:
            self.camera.stop()

    def update_camera(self):
        now = time.monotonic()
        if self.capture_job is not None and self.capture_job.done():
            token, captured_at, observation, preview, status = self.capture_job.result()
            self.capture_job = None
            if self.session.observe(token, observation, captured_at):
                self.preview = preview
                self.camera_status = status
        token = self.session.sample_token
        if token and self.capture_job is None and now >= self.next_sample:
            self.capture_job = self.camera.executor.submit(self.camera.read, token)
            self.next_sample = now + 0.25

    def click(self, pos):
        if self.camera_button.collidepoint(pos):
            self.toggle_camera()
        elif self.restart_button.collidepoint(pos):
            self.session.restart()
        elif self.undo_button.collidepoint(pos):
            self.session.undo()
        else:
            x, y = BOARD_ORIGIN
            col = (pos[0] - x + CELL // 2) // CELL
            row = (pos[1] - y + CELL // 2) // CELL
            self.session.move(row, col)

    def text(self, text, x, y, color=(233, 236, 222), title=False):
        font = self.title if title else self.font
        self.screen.blit(font.render(text, True, color), (x, y))

    def button(self, rect, label, enabled=True):
        color = (54, 79, 64) if enabled else (52, 60, 54)
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        self.text(label, rect.x + 12, rect.y + 6)

    def draw(self):
        surface = self.screen
        surface.fill((24, 31, 27))
        self.text("五子棋", 27, 15, title=True)
        self.text("你执黑先行，先连成五枚或更多同色棋子获胜。", 124, 25)
        x, y = BOARD_ORIGIN
        span = (BOARD_SIZE - 1) * CELL
        pygame.draw.rect(surface, (221, 172, 109), (x - 19, y - 19, span + 38, span + 38), border_radius=10)
        for index in range(BOARD_SIZE):
            delta = index * CELL
            pygame.draw.line(surface, (108, 73, 46), (x, y + delta), (x + span, y + delta))
            pygame.draw.line(surface, (108, 73, 46), (x + delta, y), (x + delta, y + span))
        for row, col in ((3, 3), (3, 11), (7, 7), (11, 3), (11, 11)):
            pygame.draw.circle(surface, (108, 73, 46), (x + col * CELL, y + row * CELL), 4)
        game = self.session.game
        last = game.history[-1] if game.history else None
        for row, cells in enumerate(game.board):
            for col, piece in enumerate(cells):
                if piece:
                    point = (x + col * CELL, y + row * CELL)
                    pygame.draw.circle(surface, (31, 32, 29) if piece == BLACK else (246, 240, 220), point, 15)
                    if (row, col) == last:
                        pygame.draw.circle(surface, (207, 112, 72), point, 4)
        if game.winning_line:
            first, final = game.winning_line[0], game.winning_line[-1]
            pygame.draw.line(surface, (189, 67, 48), (x + first[1] * CELL, y + first[0] * CELL),
                             (x + final[1] * CELL, y + final[0] * CELL), 3)

        if game.game_over:
            status = "你赢了！" if game.winner == BLACK else "白棋获胜" if game.winner else "和棋"
        else:
            status = "轮到你落子" if self.session.player_turn else "AI 正在思考…"
        self.text(status, 629, 64, title=True)
        self.text("AI 难度 · " + DIFFICULTIES[self.session.policy.difficulty], 629, 108)
        self.text(DECISIONS.get(self.session.decision, DECISIONS["unchanged"]), 629, 134)
        self.button(self.restart_button, "新局  R")
        self.button(self.undo_button, "悔棋  Z", self.session.can_undo)
        self.button(self.camera_button, ("关闭" if self.session.camera_enabled else "开启") + "表情辅助  C")
        self.text("图像由本机处理。", 629, 270, (185, 198, 185))
        self.text(self.camera_status, 629, 298)
        if self.preview is not None:
            surface.blit(pygame.surfarray.make_surface(self.preview.swapaxes(0, 1)), (629, 332))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        try:
            while True:
                self.session.tick()
                self.update_camera()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.click(event.pos)
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            return
                        if event.key == pygame.K_r:
                            self.session.restart()
                        if event.key == pygame.K_z:
                            self.session.undo()
                        if event.key == pygame.K_c:
                            self.toggle_camera()
                self.draw()
                clock.tick(60)
        finally:
            self.camera.close()
            self.session.close()
            pygame.quit()


if __name__ == "__main__":
    pygame.init()
    DesktopApp().run()
