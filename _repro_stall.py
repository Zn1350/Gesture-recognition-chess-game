import sys, time
import ai_engine
ai_engine._score_to_ai_params = lambda skill_score: (1, 0.0, 0.1, 0.0)
from chess_game import ChessGame, RED, BLACK
from ai_engine import AIEngine

# Reproduce experiment.py game 2 of A-vs-B pairing: engine2(disable_nn=False) plays RED, engine1(disable_nn=True) plays BLACK
engine_red = AIEngine(disable_nn=False)   # B组
engine_black = AIEngine(disable_nn=True)  # A组

g = ChessGame()
for ply in range(30):
    color = g.current_turn
    eng = engine_red if color == RED else engine_black
    t0 = time.time()
    mv = eng.get_move(g, 80, color=color)
    dt = time.time() - t0
    print(f"ply {ply+1} color={color} move={mv} took {dt:.2f}s", flush=True)
    if mv is None:
        print("no move, game over, winner=", g.winner)
        break
    ok = g.make_move(*mv)
    if not ok:
        print("ILLEGAL MOVE RETURNED", mv)
        break
    if g.game_over:
        print("game_over winner=", g.winner)
        break
