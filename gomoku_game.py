"""Rules for a 15 × 15 freestyle Gomoku game."""

from collections.abc import Sequence

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = -1
CONNECT = 5
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

Move = tuple[int, int]
Board = Sequence[Sequence[int]]


def line_extent(
    board: Board, row: int, col: int, side: int, dr: int, dc: int
) -> tuple[int, int, int]:
    """Count connected stones and open ends, treating the center as `side`."""
    size = len(board)
    lengths = []
    open_ends = 0
    for sign in (-1, 1):
        step_row, step_col = sign * dr, sign * dc
        r, c = row + step_row, col + step_col
        length = 0
        while 0 <= r < size and 0 <= c < size and board[r][c] == side:
            length += 1
            r += step_row
            c += step_col
        if 0 <= r < size and 0 <= c < size and board[r][c] == EMPTY:
            open_ends += 1
        lengths.append(length)
    return lengths[0], lengths[1], open_ends


class GomokuGame:
    def __init__(self):
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.history: list[Move] = []
        self.current_turn = BLACK
        self.game_over = False
        self.winner: int | None = None
        self.winning_line: tuple[Move, ...] = ()

    def board_copy(self) -> tuple[tuple[int, ...], ...]:
        """Return an immutable snapshot suitable for a background search."""
        return tuple(tuple(row) for row in self.board)

    def make_move(self, row: int, col: int) -> bool:
        if (
            self.game_over
            or type(row) is not int
            or type(col) is not int
            or not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE)
            or self.board[row][col] != EMPTY
        ):
            return False

        side = self.current_turn
        self.board[row][col] = side
        self.history.append((row, col))
        for dr, dc in DIRECTIONS:
            before, after, _ = line_extent(self.board, row, col, side, dr, dc)
            if before + 1 + after >= CONNECT:
                self.winning_line = tuple(
                    (row + step * dr, col + step * dc)
                    for step in range(-before, after + 1)
                )
                self.winner = side
                self.game_over = True
                return True

        self.game_over = len(self.history) == BOARD_SIZE ** 2
        if not self.game_over:
            self.current_turn = -side
        return True

    def undo_last_turn(self) -> bool:
        """Undo the last black move and its white reply, if present."""
        if not self.history:
            return False
        plies = 1 if len(self.history) % 2 else 2
        for _ in range(plies):
            row, col = self.history.pop()
            self.board[row][col] = EMPTY
        self.current_turn = BLACK
        self.game_over = False
        self.winner = None
        self.winning_line = ()
        return True
