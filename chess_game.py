# -*- coding: utf-8 -*-
"""
中国象棋游戏逻辑模块
负责棋盘状态管理、棋子合法走棋规则校验、将军/将死判断
"""

from __future__ import annotations
from collections import deque
from copy import deepcopy
from typing import List, Optional, Tuple

# 最近局面历史窗口长度：AI 根节点选招时用它识别"会导致回到最近出现过
# 局面"的招法并降权，抑制无历史感知的确定性搜索反复选中同一组"来回走"
# 招法、陷入死循环导致假平局。12 步覆盖常见的 2~3 回合来回震荡模式。
RECENT_KEYS_MAXLEN = 12

# 棋子类型常量
EMPTY  = 0
# 红方（大写）
R_KING   = 1   # 帅
R_GUARD  = 2   # 仕
R_BISHOP = 3   # 相
R_ROOK   = 4   # 车
R_KNIGHT = 5   # 马
R_CANNON = 6   # 炮
R_PAWN   = 7   # 兵
# 黑方（负值）
B_KING   = -1  # 将
B_GUARD  = -2  # 士
B_BISHOP = -3  # 象
B_ROOK   = -4  # 车
B_KNIGHT = -5  # 马
B_CANNON = -6  # 炮
B_PAWN   = -7  # 卒

RED   = 1
BLACK = -1

# 棋子显示字符
PIECE_CHARS = {
    R_KING: '帅', R_GUARD: '仕', R_BISHOP: '相',
    R_ROOK: '车', R_KNIGHT: '马', R_CANNON: '炮', R_PAWN: '兵',
    B_KING: '将', B_GUARD: '士', B_BISHOP: '象',
    B_ROOK: '車', B_KNIGHT: '馬', B_CANNON: '砲', B_PAWN: '卒',
}

# 棋子分值（用于 AI 评估）
PIECE_VALUES = {
    R_KING: 10000, R_GUARD: 200, R_BISHOP: 200,
    R_ROOK: 1000,  R_KNIGHT: 450, R_CANNON: 450, R_PAWN: 100,
    B_KING: -10000, B_GUARD: -200, B_BISHOP: -200,
    B_ROOK: -1000,  B_KNIGHT: -450, B_CANNON: -450, B_PAWN: -100,
}

# 初始棋盘布局（黑方在上方 rows 0-4，红方在下方 rows 5-9）
INITIAL_BOARD = [
    [B_ROOK, B_KNIGHT, B_BISHOP, B_GUARD, B_KING, B_GUARD, B_BISHOP, B_KNIGHT, B_ROOK],
    [EMPTY,  EMPTY,    EMPTY,    EMPTY,   EMPTY,  EMPTY,   EMPTY,    EMPTY,    EMPTY ],
    [EMPTY,  B_CANNON, EMPTY,    EMPTY,   EMPTY,  EMPTY,   EMPTY,    B_CANNON, EMPTY ],
    [B_PAWN, EMPTY,    B_PAWN,   EMPTY,   B_PAWN, EMPTY,   B_PAWN,   EMPTY,    B_PAWN],
    [EMPTY,  EMPTY,    EMPTY,    EMPTY,   EMPTY,  EMPTY,   EMPTY,    EMPTY,    EMPTY ],
    [EMPTY,  EMPTY,    EMPTY,    EMPTY,   EMPTY,  EMPTY,   EMPTY,    EMPTY,    EMPTY ],
    [R_PAWN, EMPTY,    R_PAWN,   EMPTY,   R_PAWN, EMPTY,   R_PAWN,   EMPTY,    R_PAWN],
    [EMPTY,  R_CANNON, EMPTY,    EMPTY,   EMPTY,  EMPTY,   EMPTY,    R_CANNON, EMPTY ],
    [EMPTY,  EMPTY,    EMPTY,    EMPTY,   EMPTY,  EMPTY,   EMPTY,    EMPTY,    EMPTY ],
    [R_ROOK, R_KNIGHT, R_BISHOP, R_GUARD, R_KING, R_GUARD, R_BISHOP, R_KNIGHT, R_ROOK],
]


def _color(piece: int) -> Optional[int]:
    if piece > 0: return RED
    if piece < 0: return BLACK
    return None


def _is_in_bounds(r: int, c: int) -> bool:
    return 0 <= r <= 9 and 0 <= c <= 8


def _in_palace(r: int, c: int, color: int) -> bool:
    """检查位置是否在对应方九宫格内"""
    if color == RED:
        return 7 <= r <= 9 and 3 <= c <= 5
    else:
        return 0 <= r <= 2 and 3 <= c <= 5


class ChessGame:
    def __init__(self):
        self.board: List[List[int]] = [row[:] for row in INITIAL_BOARD]
        self.current_turn: int = RED
        self.move_history: List[Tuple] = []
        self.captured_pieces: List[int] = []
        self.game_over: bool = False
        self.winner: Optional[int] = None
        # 最近局面历史（含当前局面），用于 AI 根节点招法震荡检测，见 RECENT_KEYS_MAXLEN
        self.recent_keys: deque = deque([self.board_key()], maxlen=RECENT_KEYS_MAXLEN)

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def board_key(self) -> Tuple:
        """当前局面的可哈希 key（棋盘 + 走棋方），用于置换表 / 重复局面检测"""
        return (tuple(tuple(row) for row in self.board), self.current_turn)

    def get_piece(self, r: int, c: int) -> int:
        return self.board[r][c]

    def get_legal_moves(self, r: int, c: int) -> List[Tuple[int, int]]:
        """返回 (r,c) 处棋子所有合法目标格，已过滤走后自将局面"""
        piece = self.board[r][c]
        if piece == EMPTY or _color(piece) != self.current_turn:
            return []
        candidates = self._raw_moves(r, c, piece)
        return [m for m in candidates if not self._move_causes_self_check(r, c, m[0], m[1])]

    def make_move(self, fr: int, fc: int, tr: int, tc: int) -> bool:
        """执行走棋，成功返回 True"""
        if (tr, tc) not in self.get_legal_moves(fr, fc):
            return False
        captured = self.board[tr][tc]
        self.move_history.append((fr, fc, tr, tc, captured))
        if captured != EMPTY:
            self.captured_pieces.append(captured)
        self.board[tr][tc] = self.board[fr][fc]
        self.board[fr][fc] = EMPTY
        self.current_turn = -self.current_turn
        self.recent_keys.append(self.board_key())
        self._check_game_over()
        return True

    def undo_move(self) -> bool:
        """撤销最后一步，成功返回 True"""
        if not self.move_history:
            return False
        fr, fc, tr, tc, captured = self.move_history.pop()
        self.board[fr][fc] = self.board[tr][tc]
        self.board[tr][tc] = captured
        if captured != EMPTY and self.captured_pieces:
            self.captured_pieces.pop()
        self.current_turn = -self.current_turn
        if len(self.recent_keys) > 1:
            self.recent_keys.pop()
        self.game_over = False
        self.winner = None
        return True

    def undo_two_moves(self) -> bool:
        """悔棋：撤销最近两步（AI一步 + 玩家一步），回到玩家上一手前"""
        count = 0
        for _ in range(2):
            if self.undo_move():
                count += 1
        return count > 0

    def is_in_check(self, color: int) -> bool:
        """判断 color 方是否被将军"""
        king_pos = self._find_king(color)
        if king_pos is None:
            return True
        kr, kc = king_pos
        # 枚举对方所有棋子的伪合法走法，看能否攻击到王
        for r in range(10):
            for c in range(9):
                piece = self.board[r][c]
                if _color(piece) == -color:
                    if (kr, kc) in self._raw_moves(r, c, piece):
                        return True
        return False

    def get_all_moves(self, color: int) -> List[Tuple[int, int, int, int]]:
        """获取 color 方所有合法走法（AI 用）"""
        moves = []
        saved_turn = self.current_turn
        self.current_turn = color
        for r in range(10):
            for c in range(9):
                if _color(self.board[r][c]) == color:
                    for tr, tc in self.get_legal_moves(r, c):
                        moves.append((r, c, tr, tc))
        self.current_turn = saved_turn
        return moves

    # ── 内部规则 ──────────────────────────────────────────────────────────────

    def _raw_moves(self, r: int, c: int, piece: int) -> List[Tuple[int, int]]:
        """生成伪合法走法（不检查自将），按棋子类型分派"""
        t = abs(piece)
        color = _color(piece)
        if t == abs(R_KING):    return self._king_moves(r, c, color)
        if t == abs(R_GUARD):   return self._guard_moves(r, c, color)
        if t == abs(R_BISHOP):  return self._bishop_moves(r, c, color)
        if t == abs(R_ROOK):    return self._rook_moves(r, c, color)
        if t == abs(R_KNIGHT):  return self._knight_moves(r, c, color)
        if t == abs(R_CANNON):  return self._cannon_moves(r, c, color)
        if t == abs(R_PAWN):    return self._pawn_moves(r, c, color)
        return []

    def _king_moves(self, r, c, color):
        moves = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if _in_palace(nr, nc, color) and _color(self.board[nr][nc]) != color:
                moves.append((nr, nc))
        # 将帅照面规则
        opp_king = self._find_king(-color)
        if opp_king and opp_king[1] == c:
            blocked = any(self.board[rr][c] != EMPTY
                          for rr in range(min(r, opp_king[0])+1, max(r, opp_king[0])))
            if not blocked:
                moves.append(opp_king)
        return list(set(moves))

    def _guard_moves(self, r, c, color):
        moves = []
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            if _in_palace(nr, nc, color) and _color(self.board[nr][nc]) != color:
                moves.append((nr, nc))
        return moves

    def _bishop_moves(self, r, c, color):
        moves = []
        for dr, dc in [(-2,-2),(-2,2),(2,-2),(2,2)]:
            nr, nc = r+dr, c+dc
            # 象不能过河
            if color == RED and nr < 5: continue
            if color == BLACK and nr > 4: continue
            if not _is_in_bounds(nr, nc): continue
            # 检查象眼
            er, ec = r+dr//2, c+dc//2
            if self.board[er][ec] == EMPTY and _color(self.board[nr][nc]) != color:
                moves.append((nr, nc))
        return moves

    def _rook_moves(self, r, c, color):
        moves = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            while _is_in_bounds(nr, nc):
                target = self.board[nr][nc]
                if target == EMPTY:
                    moves.append((nr, nc))
                elif _color(target) != color:
                    moves.append((nr, nc))
                    break
                else:
                    break
                nr, nc = nr+dr, nc+dc
        return moves

    def _knight_moves(self, r, c, color):
        moves = []
        # 马走"日"字，先直后斜，需检查马腿
        for dr, dc, lr, lc in [
            (-2,-1,-1,0),(-2,1,-1,0),
            (2,-1,1,0),(2,1,1,0),
            (-1,-2,0,-1),(-1,2,0,1),
            (1,-2,0,-1),(1,2,0,1),
        ]:
            nr, nc = r+dr, c+dc
            if not _is_in_bounds(nr, nc): continue
            if self.board[r+lr][c+lc] != EMPTY: continue  # 马腿被挡
            if _color(self.board[nr][nc]) != color:
                moves.append((nr, nc))
        return moves

    def _cannon_moves(self, r, c, color):
        moves = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            # 不吃子：沿路空格移动
            while _is_in_bounds(nr, nc) and self.board[nr][nc] == EMPTY:
                moves.append((nr, nc))
                nr, nc = nr+dr, nc+dc
            # 越过一个棋子后吃对方棋子（炮架）
            if _is_in_bounds(nr, nc):
                nr, nc = nr+dr, nc+dc
                while _is_in_bounds(nr, nc):
                    target = self.board[nr][nc]
                    if target != EMPTY:
                        if _color(target) != color:
                            moves.append((nr, nc))
                        break
                    nr, nc = nr+dr, nc+dc
        return moves

    def _pawn_moves(self, r, c, color):
        moves = []
        forward = -1 if color == RED else 1
        # 向前
        nr, nc = r+forward, c
        if _is_in_bounds(nr, nc) and _color(self.board[nr][nc]) != color:
            moves.append((nr, nc))
        # 过河后可左右移动
        crossed = (color == RED and r <= 4) or (color == BLACK and r >= 5)
        if crossed:
            for dc in [-1, 1]:
                nr, nc = r, c+dc
                if _is_in_bounds(nr, nc) and _color(self.board[nr][nc]) != color:
                    moves.append((nr, nc))
        return moves

    def _find_king(self, color: int) -> Optional[Tuple[int, int]]:
        target = R_KING if color == RED else B_KING
        for r in range(10):
            for c in range(9):
                if self.board[r][c] == target:
                    return (r, c)
        return None

    def _move_causes_self_check(self, fr, fc, tr, tc) -> bool:
        """走这步后己方是否被将军"""
        saved = self.board[tr][tc]
        self.board[tr][tc] = self.board[fr][fc]
        self.board[fr][fc] = EMPTY
        in_check = self.is_in_check(self.current_turn)
        self.board[fr][fc] = self.board[tr][tc]
        self.board[tr][tc] = saved
        return in_check

    def _check_game_over(self):
        opponent = self.current_turn
        if not self.get_all_moves(opponent):
            self.game_over = True
            self.winner = -opponent

    def board_copy(self) -> 'ChessGame':
        """深拷贝，供 AI 搜索使用（不拷贝历史以节省内存）"""
        g = ChessGame.__new__(ChessGame)
        g.board = [row[:] for row in self.board]
        g.current_turn = self.current_turn
        g.move_history = []
        g.captured_pieces = []
        g.game_over = self.game_over
        g.winner = self.winner
        # 搜索用副本不需要重复局面历史：震荡检测只在真实对局的根节点做一次，
        # 不随每个模拟节点传播，保持搜索开销不变。给空 deque 只是为了让
        # board_key()/recent_keys 属性始终存在，避免误用时报 AttributeError。
        g.recent_keys = deque(maxlen=RECENT_KEYS_MAXLEN)
        return g
