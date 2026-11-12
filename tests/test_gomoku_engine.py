import unittest
from unittest.mock import patch

from gomoku_ai import PROFILES, DifficultyProfile, GomokuAI
from gomoku_game import BLACK, BOARD_SIZE, EMPTY, WHITE, GomokuGame


class GomokuEngineTests(unittest.TestCase):
    def setUp(self):
        self.game = GomokuGame()
        self.ai = GomokuAI()

    def play(self, moves):
        for move in moves:
            self.assertTrue(self.game.make_move(*move), move)

    def test_rejected_moves_do_not_change_the_board_or_turn(self):
        self.play([(7, 7)])
        snapshot = self.game.board_copy()
        for move in [(7, 7), (-1, 7), (0, BOARD_SIZE), (True, 3), (1.2, 3)]:
            with self.subTest(move=move):
                self.assertFalse(self.game.make_move(*move))
                self.assertEqual(self.game.current_turn, WHITE)
                self.assertEqual(self.game.board_copy(), snapshot)

    def test_overline_wins_and_undo_restores_black_turn(self):
        self.play([
            (0, 3), (13, 0), (1, 3), (13, 2), (2, 3), (13, 4),
            (4, 3), (13, 6), (5, 3), (13, 8), (3, 3),
        ])
        self.assertTrue(self.game.game_over)
        self.assertEqual(self.game.winner, BLACK)
        self.assertEqual(self.game.winning_line, tuple((row, 3) for row in range(6)))
        self.assertFalse(self.game.make_move(8, 8))
        self.assertTrue(self.game.undo_last_turn())
        self.assertFalse(self.game.game_over)
        self.assertIsNone(self.game.winner)
        self.assertEqual(self.game.current_turn, BLACK)
        self.assertEqual(self.game.board[3][3], EMPTY)
        self.assertEqual(len(self.game.history), 10)

    def test_white_antidiagonal_win_and_pair_undo(self):
        self.play([
            (9, 0), (0, 4), (9, 2), (1, 3), (9, 4), (2, 2),
            (9, 6), (3, 1), (9, 8), (4, 0),
        ])
        self.assertEqual(self.game.winner, WHITE)
        self.assertEqual(self.game.winning_line, tuple((row, 4 - row) for row in range(5)))
        self.assertTrue(self.game.undo_last_turn())
        self.assertEqual(self.game.current_turn, BLACK)
        self.assertEqual(self.game.board[9][8], EMPTY)
        self.assertEqual(self.game.board[4][0], EMPTY)
        self.assertEqual(len(self.game.history), 8)

    def test_full_board_without_five_is_a_draw(self):
        # Pairs in each row alternate, and every second row reverses colors.
        black, white = [], []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                side = white if (row + col // 2) % 2 else black
                side.append((row, col))
        for black_move, white_move in zip(black, white):
            self.play([black_move, white_move])
        self.play([black[-1]])
        self.assertTrue(self.game.game_over)
        self.assertIsNone(self.game.winner)
        self.assertIsNone(self.ai.choose_move(self.game.board_copy()).move)

    def test_immediate_win_precedes_a_block_at_every_level(self):
        self.play([
            (0, 0), (4, 7), (0, 1), (4, 8), (0, 2), (4, 9),
            (0, 3), (4, 10), (14, 14),
        ])
        for level in PROFILES:
            with self.subTest(level=level):
                result = self.ai.choose_move(self.game.board_copy(), level=level)
                self.assertEqual(result.reason, "win")
                self.assertIn(result.move, [(4, 6), (4, 11)])
                self.assertEqual(result.nodes, 0)

    def test_mandatory_block_is_identical_at_every_level(self):
        # White can create a fork, but black would win before white can finish it.
        self.play([(0, 0), (6, 5), (1, 1), (6, 6), (3, 3), (6, 7), (4, 4)])
        for level in PROFILES:
            with self.subTest(level=level):
                result = self.ai.choose_move(self.game.board_copy(), level=level)
                self.assertEqual((result.move, result.reason), ((2, 2), "block"))
                self.assertEqual(result.nodes, 0)

    def test_hidden_fork_is_found_before_candidate_pruning(self):
        # White's unique fork ranked 39th before this check, below every search width.
        self.play([
            (6, 11), (7, 8), (8, 4), (6, 8), (9, 3), (6, 5), (7, 3),
            (4, 11), (8, 5), (8, 8), (9, 10), (5, 10), (8, 7), (4, 7),
            (9, 8), (9, 5), (3, 6), (4, 10), (5, 3),
        ])
        snapshot = self.game.board_copy()
        for level in PROFILES:
            with self.subTest(level=level):
                result = self.ai.choose_move(self.game.board, level=level)
                self.assertEqual((result.move, result.reason), ((4, 8), "fork"))
                self.assertEqual(result.nodes, 0)
        self.assertEqual(self.game.board_copy(), snapshot)

        # The two winning replies are independent. Blocking either leaves the other.
        self.play([(4, 8)])
        for block, reply in [((4, 9), (5, 8)), ((5, 8), (4, 9))]:
            self.play([block, reply])
            self.assertEqual(self.game.winner, WHITE)
            self.assertTrue(self.game.undo_last_turn())

    def test_counter_threat_defends_independent_enemy_forks(self):
        self.play([
            (2, 5), (0, 0), (2, 6), (0, 14), (2, 7), (6, 0), (7, 5),
            (6, 14), (7, 6), (14, 0), (7, 7), (12, 5), (12, 3), (12, 6),
            (12, 9), (12, 7), (14, 14),
        ])
        # Black has forks on two rows. White can use its own four to force a reply.
        for level in PROFILES:
            with self.subTest(level=level):
                result = self.ai.choose_move(self.game.board_copy(), level=level)
                self.assertEqual(result.reason, "defend_fork")
                self.assertIn(result.move, [(12, 4), (12, 8)])

    def test_same_position_uses_distinct_search_budgets(self):
        self.play([
            (4, 5), (8, 10), (5, 11), (4, 6), (9, 8), (6, 9), (6, 11),
            (6, 10), (11, 10), (8, 11), (4, 8), (6, 8), (9, 11),
        ])
        snapshot = self.game.board_copy()
        decisions = [self.ai.choose_move(self.game.board, level=level) for level in PROFILES]
        self.assertEqual(self.game.board_copy(), snapshot)
        self.assertEqual([result.completed_depth for result in decisions], [1, 2, 3])
        self.assertLess(decisions[0].nodes, decisions[1].nodes)
        self.assertLess(decisions[1].nodes, decisions[2].nodes)
        for result in decisions:
            self.assertEqual(result.reason, "search")
            self.assertLessEqual(result.nodes, result.profile.node_budget)
            row, col = result.move
            self.assertEqual(snapshot[row][col], EMPTY)

        self.assertEqual(len({result.move for result in decisions}), len(PROFILES))

    def test_exhausted_search_falls_back_without_mutating_input(self):
        self.play([(7, 7)])
        snapshot = self.game.board_copy()
        first = self.ai.choose_move(snapshot, level=1)
        with patch.dict(PROFILES, {3: DifficultyProfile(3, depth=3, width=8, node_budget=2)}):
            limited = self.ai.choose_move(self.game.board, level=3)
        self.assertEqual(limited.move, first.move)
        self.assertEqual(limited.completed_depth, 1)
        self.assertEqual(limited.nodes, 2)
        self.assertEqual(self.game.board_copy(), snapshot)


if __name__ == "__main__":
    unittest.main()
