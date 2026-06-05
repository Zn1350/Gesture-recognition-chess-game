# -*- coding: utf-8 -*-
"""
网页版康复象棋 — Flask 后端
复用 chess_game / assessment / ai_engine / gesture_rl 模块，
通过 JSON API 与前端通信。手势识别在浏览器前端（MediaPipe JS）完成，
后端只负责棋局逻辑、康复评估和 RL 阈值调整。
"""

import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, request, render_template
from chess_game import ChessGame, RED, BLACK, PIECE_CHARS
from assessment import AssessmentModule
from ai_engine import AIEngine
from gesture_rl import GestureRL
import config

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── 全局状态（单用户） ────────────────────────────────────────────────────────
_lock         = threading.Lock()
_game         = ChessGame()
_assessment   = AssessmentModule()
_ai           = AIEngine()
_gesture_rl   = GestureRL(
    init_pinch   = config.GESTURE_PINCH_THRESHOLD,
    init_release = config.GESTURE_RELEASE_THRESHOLD,
    init_hold    = config.GESTURE_MIN_HOLD_FRAMES,
)
_ai_thinking      = False
_selected_pos     = None
_legal_moves      = []
_scissors_enabled = False
_settings         = {
    "use_palm": config.GESTURE_USE_PALM,
    "smooth_alpha": getattr(config, 'GESTURE_DISPLAY_ALPHA', 0.45),
    "smooth_win": config.GESTURE_SMOOTH_WINDOW,
}


def _board_json(game: ChessGame):
    cells = []
    for r in range(10):
        row = []
        for c in range(9):
            p = game.board[r][c]
            row.append({
                "piece": p,
                "char":  PIECE_CHARS.get(p, ""),
                "color": "red" if p > 0 else ("black" if p < 0 else ""),
            })
        cells.append(row)
    return cells


def _state_json():
    global _selected_pos, _legal_moves, _ai_thinking, _scissors_enabled
    stats   = _assessment.get_stats()
    rl_stats = _gesture_rl.get_stats()
    return {
        "board":        _board_json(_game),
        "current_turn": "red" if _game.current_turn == RED else "black",
        "game_over":    _game.game_over,
        "winner":       ("red" if _game.winner == RED else "black") if _game.winner else None,
        "selected":     list(_selected_pos) if _selected_pos else None,
        "legal_moves":  [list(m) for m in _legal_moves],
        "ai_thinking":  _ai_thinking,
        "scissors_enabled": _scissors_enabled,
        "rl_thresholds": {
            "pinch":   round(rl_stats.pinch_threshold, 4),
            "release": round(rl_stats.release_threshold, 4),
            "hold":    rl_stats.hold_frames,
        },
        "stats": {
            "skill_score":    round(stats.skill_score, 1),
            "skill_label":    stats.skill_label,
            "total_moves":    stats.total_moves,
            "captures":       stats.captures,
            "error_rate":     round(stats.error_rate * 100, 1),
            "avg_move_time":  round(stats.avg_move_time, 1),
            "ai_blunder_rate":round(stats.ai_blunder_rate * 100, 1),
            "ai_search_depth":stats.ai_search_depth,
            "last_ql_reward": round(stats.last_ql_reward, 2),
        },
        "settings": _settings,
    }


# ── 路由 ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify(_state_json())


@app.route("/api/grab", methods=["POST"])
def api_grab():
    """GRABBING 时：选子（仅每次进入新格子调用一次）"""
    global _selected_pos, _legal_moves
    data = request.get_json()
    r, c = int(data["row"]), int(data["col"])

    with _lock:
        if _game.game_over or _ai_thinking or _game.current_turn != RED:
            return jsonify(_state_json())

        piece = _game.board[r][c]
        if piece > 0:
            # 有己方棋子 → 选中（允许切换到另一颗）
            _selected_pos = (r, c)
            _legal_moves  = _game.get_legal_moves(r, c)
            _assessment.start_move_timer()
            _gesture_rl.record_grab_success()
        else:
            # 无己方棋子 → grab miss（只在无选中时记录）
            if _selected_pos is None:
                _gesture_rl.record_grab_miss()
                _assessment.record_illegal_attempt()

        return jsonify(_state_json())


@app.route("/api/release", methods=["POST"])
def api_release():
    """RELEASING 时：落子"""
    global _selected_pos, _legal_moves, _scissors_enabled
    data = request.get_json()
    r, c = int(data["row"]), int(data["col"])

    with _lock:
        if _game.game_over or _ai_thinking or _game.current_turn != RED:
            return jsonify(_state_json())

        if _selected_pos is None:
            _gesture_rl.record_release_miss()
            _assessment.record_illegal_attempt()
            return jsonify(_state_json())

        target = (r, c)

        if target == _selected_pos:
            # 松开在同一格 → 取消选中
            _selected_pos = None
            _legal_moves  = []
            return jsonify(_state_json())

        if target in _legal_moves:
            fr, fc = _selected_pos
            captured = _game.board[r][c] != 0
            _game.make_move(fr, fc, r, c)
            _assessment.record_move(captured=captured)
            _gesture_rl.record_release_success()
            _selected_pos     = None
            _legal_moves      = []
            _scissors_enabled = True
            if _game.game_over:
                _assessment.record_game_result(player_won=(_game.winner == RED))
            else:
                _trigger_ai()
        else:
            # 非法落点
            piece = _game.board[r][c]
            if piece > 0:
                # 松开到己方另一颗棋子 → 切换选中
                _selected_pos = (r, c)
                _legal_moves  = _game.get_legal_moves(r, c)
                _assessment.start_move_timer()
                _gesture_rl.record_grab_success()
            else:
                _gesture_rl.record_release_miss()
                _assessment.record_illegal_attempt()
                _selected_pos = None
                _legal_moves  = []

        return jsonify(_state_json())


@app.route("/api/release_miss", methods=["POST"])
def api_release_miss():
    """RELEASING 时落在棋盘外"""
    with _lock:
        _gesture_rl.record_release_miss()
        _assessment.record_illegal_attempt()
        return jsonify(_state_json())


@app.route("/api/dwell_timeout", methods=["POST"])
def api_dwell_timeout():
    """手在某格停留超过 3 秒未操作"""
    with _lock:
        _gesture_rl.record_dwell_timeout()
        return jsonify(_state_json())


@app.route("/api/undo", methods=["POST"])
def api_undo():
    global _selected_pos, _legal_moves, _ai_thinking, _scissors_enabled
    with _lock:
        if _ai_thinking or _game.game_over:
            return jsonify(_state_json())
        _game.undo_two_moves()
        _assessment.record_illegal_attempt()
        _gesture_rl.record_release_miss()
        _selected_pos     = None
        _legal_moves      = []
        _scissors_enabled = False
        return jsonify(_state_json())


@app.route("/api/restart", methods=["POST"])
def api_restart():
    global _game, _assessment, _selected_pos, _legal_moves, _ai_thinking, _scissors_enabled
    global _gesture_rl
    with _lock:
        _game       = ChessGame()
        _assessment = AssessmentModule()
        _gesture_rl._grab_window.clear()
        _gesture_rl._release_window.clear()
        _gesture_rl._all_window.clear()
        _gesture_rl._success_streak = 0
        _selected_pos     = None
        _legal_moves      = []
        _ai_thinking      = False
        _scissors_enabled = False
        return jsonify(_state_json())


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """获取或更新开发设置（use_palm, smooth_alpha, smooth_win）。"""
    global _settings
    if request.method == 'GET':
        with _lock:
            return jsonify(_settings)

    # POST -> 更新
    data = request.get_json() or {}
    with _lock:
        if 'use_palm' in data:
            _settings['use_palm'] = bool(data['use_palm'])
        if 'smooth_alpha' in data:
            try:
                _settings['smooth_alpha'] = float(data['smooth_alpha'])
            except Exception:
                pass
        if 'smooth_win' in data:
            try:
                _settings['smooth_win'] = int(data['smooth_win'])
            except Exception:
                pass
        return jsonify(_settings)


# ── AI 异步走棋 ───────────────────────────────────────────────────────────────

def _trigger_ai():
    global _ai_thinking
    _ai_thinking = True

    def _work():
        global _ai_thinking, _scissors_enabled
        time.sleep(0.4)
        with _lock:
            if _game.game_over or _game.current_turn != BLACK:
                _ai_thinking = False
                return
            stats_snap = _assessment.get_stats()
            game_copy  = _game.board_copy()
            move = _ai.get_move(game_copy, stats_snap.skill_score, stats=stats_snap)
            if move and not _game.game_over:
                fr, fc, tr, tc = move
                captured = _game.board[tr][tc] != 0
                _game.make_move(fr, fc, tr, tc)
                _assessment.record_move(captured=captured)
                if _game.game_over:
                    _assessment.record_game_result(player_won=(_game.winner == RED))
            _ai_thinking = False

    threading.Thread(target=_work, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
