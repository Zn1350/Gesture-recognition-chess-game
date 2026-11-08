import threading
import time
import unittest
from unittest.mock import patch

from emotion import ExpressionObservation
from gomoku_session import GomokuSession
from web.app import create_app


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.session = GomokuSession()
        self.addCleanup(self.session.close)

    def finish_reply(self):
        deadline = time.monotonic() + 3
        while not self.session.player_turn and time.monotonic() < deadline:
            self.session.tick()
            time.sleep(0.005)
        self.assertTrue(self.session.player_turn)

    def test_turn_freezes_difficulty_and_rejects_old_evidence(self):
        face = ExpressionObservation("sad", 0.90, {"sad": 0.90, "neutral": 0.10}, 0.95)
        with patch("gomoku_session.time.monotonic", return_value=100):
            self.session.set_camera(True)
        token = self.session.sample_token
        for index in range(5):
            self.assertTrue(self.session.observe(token, face, 100 + index * 0.25))
        with patch("gomoku_session.time.monotonic", return_value=101.1):
            self.assertTrue(self.session.move(7, 7))
        self.assertEqual(self.session.decision, "lowered")
        self.assertIsNone(self.session.sample_token)
        self.assertFalse(self.session.observe(token, face, 101.2))
        self.finish_reply()
        self.assertEqual(self.session.last_ai.profile.level, 2)
        self.assertNotEqual(token, self.session.sample_token)
        self.assertFalse(self.session.observe(token, face, time.monotonic()))

    def test_restarted_game_discards_pending_ai_result(self):
        started = threading.Event()
        resume = threading.Event()
        choose_move = self.session.ai.choose_move

        def blocked_search(*args):
            started.set()
            resume.wait(3)
            return choose_move(*args)

        with patch.object(self.session.ai, "choose_move", side_effect=blocked_search):
            self.assertTrue(self.session.move(7, 7))
            old_future = self.session._future
            try:
                self.assertTrue(started.wait(3))
                self.session.restart()
            finally:
                resume.set()
            self.assertIsNotNone(old_future.result(timeout=3).move)
        self.session.tick()
        self.assertEqual(self.session.game.history, [])
        self.assertTrue(self.session.player_turn)

    def test_local_api_and_camera_gate(self):
        client = create_app(self.session).test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertIsNone(client.get("/api/state").json["sample_token"])
        self.assertEqual(client.post("/api/move", json={"row": True, "col": 0}).status_code, 400)
        self.assertEqual(client.post("/api/move", json={"row": 15, "col": 0}).status_code, 409)
        token = client.post("/api/camera", json={"enabled": True}).json["sample_token"]
        headers = {"X-Observation-Token": token}
        self.assertEqual(client.post("/api/expression", headers=headers).status_code, 415)
        self.assertEqual(client.post("/api/move", json={"row": 7, "col": 7}).status_code, 200)
        self.assertEqual(client.post("/api/expression", headers=headers).status_code, 409)
        self.finish_reply()
        self.assertEqual(client.post("/api/undo").json["board"][7][7], 0)
        self.assertIsNone(client.post("/api/camera", json={"enabled": False}).json["sample_token"])


if __name__ == "__main__":
    unittest.main()
