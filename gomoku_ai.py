"""Bounded Gomoku search with a shared tactical check for all levels."""

from dataclasses import dataclass

from gomoku_game import BLACK, BOARD_SIZE, CONNECT, DIRECTIONS, EMPTY, WHITE, Board, Move, line_extent

_MATE = 1_000_000
_FORCED_BLOCK_COST = 25_000
_NEIGHBORS = tuple(
    (dr, dc)
    for dr in range(-2, 3)
    for dc in range(-2, 3)
    if dr or dc
)
_SHAPE_WEIGHTS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 6, 26),
    (0, 55, 280),
    (0, 600, 5_000),
)


@dataclass(frozen=True)
class DifficultyProfile:
    level: int
    depth: int
    width: int
    node_budget: int


PROFILES = {
    1: DifficultyProfile(1, depth=1, width=3, node_budget=4),
    2: DifficultyProfile(2, depth=2, width=5, node_budget=32),
    3: DifficultyProfile(3, depth=3, width=8, node_budget=120),
}


@dataclass(frozen=True)
class MoveDecision:
    move: Move | None
    reason: str
    profile: DifficultyProfile
    nodes: int = 0
    completed_depth: int = 0


@dataclass(frozen=True)
class _Candidate:
    move: Move
    attack: int
    defense: int


def _shape_score(board: Board, row: int, col: int, side: int) -> int:
    score = 0
    for dr, dc in DIRECTIONS:
        before, after, open_ends = line_extent(board, row, col, side, dr, dc)
        length = before + 1 + after
        if length >= CONNECT:
            return _MATE
        score += _SHAPE_WEIGHTS[length][open_ends]
    return score


def _rank_candidates(board: Board, side: int) -> list[_Candidate]:
    size = len(board)
    frontier = set()
    for row, cells in enumerate(board):
        for col, cell in enumerate(cells):
            if cell == EMPTY:
                continue
            for dr, dc in _NEIGHBORS:
                r, c = row + dr, col + dc
                if 0 <= r < size and 0 <= c < size and board[r][c] == EMPTY:
                    frontier.add((r, c))

    candidates = [
        _Candidate((r, c), _shape_score(board, r, c, side), _shape_score(board, r, c, -side))
        for r, c in frontier
    ]
    center = size // 2
    candidates.sort(key=lambda candidate: (
        -(candidate.attack + candidate.defense),
        -candidate.attack,
        sum((coordinate - center) ** 2 for coordinate in candidate.move),
        candidate.move,
    ))
    return candidates


class _BudgetExhausted(Exception):
    pass


class _Search:
    def __init__(self, profile: DifficultyProfile):
        self.profile = profile
        self.nodes = 1  # The root position was already ranked.

    def best_move(
        self, board: list[list[int]], side: int, candidates: list[_Candidate], depth: int
    ) -> Move:
        best = candidates[0].move
        alpha = -_MATE - 1
        for candidate in candidates:
            row, col = candidate.move
            board[row][col] = side
            try:
                score = -self.score(board, -side, depth - 1, 1, -_MATE - 1, -alpha)
            finally:
                board[row][col] = EMPTY
            if score > alpha:
                alpha = score
                best = candidate.move
        return best

    def score(
        self, board: list[list[int]], side: int, remaining: int, ply: int, alpha: int, beta: int
    ) -> int:
        if self.nodes >= self.profile.node_budget:
            raise _BudgetExhausted
        self.nodes += 1

        candidates = _rank_candidates(board, side)
        if not candidates:
            return 0
        if any(candidate.attack == _MATE for candidate in candidates):
            return _MATE - ply - 1

        blocks = [candidate for candidate in candidates if candidate.defense == _MATE]
        if len(blocks) > 1:
            return -_MATE + ply + 2
        if remaining == 0:
            if blocks:
                return blocks[0].attack - _FORCED_BLOCK_COST
            return max(candidate.attack for candidate in candidates) - max(
                candidate.defense for candidate in candidates
            )

        best = -_MATE - 1
        for candidate in blocks or candidates[:self.profile.width]:
            row, col = candidate.move
            board[row][col] = side
            try:
                score = -self.score(board, -side, remaining - 1, ply + 1, -beta, -alpha)
            finally:
                board[row][col] = EMPTY
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best


class GomokuAI:
    def choose_move(self, board: Board, side: int = WHITE, level: int = 3) -> MoveDecision:
        """Choose a move on a snapshot from an active game."""
        if type(level) is not int or level not in PROFILES:
            raise ValueError("Gomoku difficulty must be 1, 2 or 3")
        if type(side) is not int or side not in (BLACK, WHITE):
            raise ValueError("Invalid Gomoku side")
        if len(board) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in board):
            raise ValueError("Expected a 15 by 15 Gomoku board")
        profile = PROFILES[level]
        if all(cell == EMPTY for row in board for cell in row):
            center = BOARD_SIZE // 2
            return MoveDecision((center, center), "opening", profile)

        candidates = _rank_candidates(board, side)
        if not candidates:
            return MoveDecision(None, "no_moves", profile)

        # A five requires adjacent stones, so the untruncated frontier contains
        # every immediate win for both sides.
        for candidate in candidates:
            if candidate.attack == _MATE:
                return MoveDecision(candidate.move, "win", profile)
        for candidate in candidates:
            if candidate.defense == _MATE:
                return MoveDecision(candidate.move, "block", profile)

        work_board = [list(row) for row in board]
        candidates = candidates[:profile.width]
        search = _Search(profile)
        best = candidates[0].move
        completed_depth = 1
        for depth in range(2, profile.depth + 1):
            try:
                best = search.best_move(work_board, side, candidates, depth)
            except _BudgetExhausted:
                break
            completed_depth = depth
            candidates.sort(key=lambda candidate: candidate.move != best)
        return MoveDecision(best, "search", profile, search.nodes, completed_depth)
