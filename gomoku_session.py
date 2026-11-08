"""Shared turn and sampling lifecycle for the desktop and local web game."""

import secrets
import time
from concurrent.futures import ThreadPoolExecutor

from emotion import DifficultyPolicy
from gomoku_ai import GomokuAI
from gomoku_game import BLACK, WHITE, GomokuGame


DIFFICULTIES = {1: "轻松", 2: "适中", 3: "标准"}
DECISIONS = {
    "initial": "本局从标准难度开始",
    "off": "表情辅助已关闭",
    "lowered": "持续出现负向表情线索，已降低一档",
    "insufficient_evidence": "有效表情样本不足，保持难度",
    "cooldown": "难度调整间隔内，保持难度",
    "minimum": "已到最低难度",
    "unchanged": "保持难度",
}


class GomokuSession:
    """Call session methods from one UI thread or under a caller-owned lock."""

    def __init__(self):
        self.ai = GomokuAI()
        self.policy = DifficultyPolicy(base_difficulty=3)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gomoku-ai")
        self.camera_enabled = False
        self.revision = 0
        self._future = None
        self.restart()

    def _new_phase(self):
        self.revision += 1
        self._phase_start = time.monotonic()
        self._token = secrets.token_hex(16) if self.camera_enabled and self.player_turn else None
        self.policy.clear_evidence()

    @property
    def player_turn(self):
        return not self.game.game_over and self.game.current_turn == BLACK

    @property
    def sample_token(self):
        return self._token

    def set_camera(self, enabled):
        if enabled != self.camera_enabled:
            self.camera_enabled = enabled
            if enabled:
                self.decision = "initial"
            else:
                self.policy.reset()
                self.decision = "off"
            self._new_phase()

    def observe(self, token, observation, captured_at):
        if token is None or token != self.sample_token or captured_at < self._phase_start:
            return False
        self.policy.observe(observation, captured_at)
        return True

    def move(self, row, col):
        if not self.player_turn or not self.game.make_move(row, col):
            return False
        if not self.game.game_over:
            if self.camera_enabled:
                decision = self.policy.decide(time.monotonic())
                self.decision = decision.reason
            # Each task owns an immutable board and the difficulty frozen for this turn.
            self._future = self._executor.submit(
                self.ai.choose_move, self.game.board_copy(), WHITE, self.policy.difficulty
            )
        self._new_phase()
        return True

    def tick(self):
        if self._future is None or not self._future.done():
            return
        future, self._future = self._future, None
        self.last_ai = future.result()
        if self.last_ai.move is not None:
            self.game.make_move(*self.last_ai.move)
        self._new_phase()

    @property
    def can_undo(self):
        return self._future is None and bool(self.game.history)

    def undo(self):
        if not self.can_undo or not self.game.undo_last_turn():
            return False
        self.last_ai = None
        self.policy.reset()
        self.decision = "initial" if self.camera_enabled else "off"
        self._new_phase()
        return True

    def restart(self):
        if self._future is not None:
            # A running search can finish, but this session no longer reads its result.
            self._future.cancel()
        self._future = None
        self.last_ai = None
        self.game = GomokuGame()
        self.policy.reset()
        self.decision = "initial" if self.camera_enabled else "off"
        self._new_phase()

    def state(self):
        return {
            "board": self.game.board,
            "black": BLACK,
            "white": WHITE,
            "current_turn": "black" if self.game.current_turn == BLACK else "white",
            "game_over": self.game.game_over,
            "winner": {BLACK: "black", WHITE: "white"}.get(self.game.winner),
            "last_move": self.game.history[-1] if self.game.history else None,
            "can_undo": self.can_undo,
            "difficulty": DIFFICULTIES[self.policy.difficulty],
            "decision": DECISIONS.get(self.decision, DECISIONS["unchanged"]),
            "sample_token": self.sample_token,
            "revision": self.revision,
        }

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
