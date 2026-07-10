# -*- coding: utf-8 -*-
"""
自我对弈数据生成脚本 —— 为 MicroDualNet 训练准备数据

用法（本地小规模烟雾测试）：
    python gen_data.py --games 4 --workers 2 --max-moves 60 --out selfplay_smoke.npz

用法（Kaggle/云端正式生成，8 进程、5 万局）：
    python gen_data.py --games 50000 --workers 8 --out selfplay_data.npz

设计要点：
  - 每个子进程内创建 AIEngine(disable_nn=True)：不加载任何神经网络，强制纯手工评估
    （_manual_evaluate），搜索深度固定为 depth（默认 2），不触发放水逻辑
    （直接调用 get_move_with_scores，绕过 get_move 的 blunder_rate 分支）。
  - 每步记录 (局面特征, 该局面所有合法走法及其原始搜索评分, 当前走棋方颜色)。
  - 对局结束（或达到 max_moves 步数上限视为和棋）得到 result（1/0/-1，红方视角），
    为每一步生成标签 z：走子方为红则 z=result，走子方为黑则 z=-result。
  - 用 multiprocessing.Pool 并行生成，进程数由 --workers 指定。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import List, Tuple

import numpy as np

from ai_engine import AIEngine
from chess_game import ChessGame, RED, BLACK
from features import extract_features


def _play_one_game(args: Tuple[int, int, int]) -> List[Tuple[np.ndarray, np.ndarray, List[int], List[Tuple[int, int, int, int]], int]]:
    """
    在子进程中对弈一局，返回该局每一步的记录列表：
      (features_vec, board_snapshot, move_scores(按 moves 顺序), moves(该局面所有合法走法), z_label)

    board_snapshot 是走这一步之前的棋盘（10x9 int8），训练时用于还原走法特征
    （_move_to_feat 需要知道起点棋子类型、目标格是否吃子，仅靠 60 维局面特征无法重建）。
    """
    _game_idx, depth, max_moves = args
    engine = AIEngine(disable_nn=True)
    game = ChessGame()

    steps: List[Tuple[np.ndarray, np.ndarray, List[int], List[Tuple[int, int, int, int]], int]] = []
    ply = 0
    while not game.game_over and ply < max_moves:
        color = game.current_turn
        feat = extract_features(game)
        board_snapshot = np.array(game.board, dtype=np.int8)
        best_move, move_scores = engine.get_move_with_scores(game, depth=depth)
        if best_move is None:
            break
        moves = list(move_scores.keys())
        scores = [move_scores[m] for m in moves]
        steps.append((feat, board_snapshot, scores, moves, color))
        game.make_move(*best_move)
        ply += 1

    if game.game_over and game.winner is not None:
        result = game.winner  # RED(1) / BLACK(-1)
    else:
        result = 0  # 达到步数上限，视为和棋

    labeled = []
    for feat, board_snapshot, scores, moves, color in steps:
        z = result if color == RED else -result
        labeled.append((feat, board_snapshot, scores, moves, z))
    return labeled


def main():
    parser = argparse.ArgumentParser(description="自我对弈生成 MicroDualNet 训练数据")
    parser.add_argument("--games", type=int, default=50000, help="总对局数")
    parser.add_argument("--workers", type=int, default=8, help="并行进程数")
    parser.add_argument("--depth", type=int, default=2, help="搜索深度")
    parser.add_argument("--max-moves", type=int, default=200, help="单局最大步数（超出视为和棋）")
    parser.add_argument("--out", type=str, default="selfplay_data.npz", help="输出文件路径")
    parser.add_argument("--progress-every", type=int, default=50, help="每完成多少局打印一次进度")
    args = parser.parse_args()

    print(f"开始生成自我对弈数据: games={args.games} workers={args.workers} depth={args.depth} max_moves={args.max_moves}")
    t_start = time.time()

    tasks = [(i, args.depth, args.max_moves) for i in range(args.games)]

    all_features: List[np.ndarray] = []
    all_boards: List[np.ndarray] = []
    all_move_scores: List[List[int]] = []
    all_moves_order: List[List[Tuple[int, int, int, int]]] = []
    all_z: List[int] = []

    completed = 0
    with mp.Pool(processes=args.workers) as pool:
        for game_steps in pool.imap_unordered(_play_one_game, tasks):
            for feat, board_snapshot, scores, moves, z in game_steps:
                all_features.append(feat)
                all_boards.append(board_snapshot)
                all_move_scores.append(scores)
                all_moves_order.append(moves)
                all_z.append(z)
            completed += 1
            if completed % args.progress_every == 0 or completed == args.games:
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (args.games - completed) / rate if rate > 0 else 0.0
                print(
                    f"[{completed}/{args.games}] 局面数={len(all_features)} "
                    f"耗时={elapsed:.1f}s 速度={rate:.2f}局/s 预估剩余={remaining/60:.1f}分钟"
                )

    features_arr = np.stack(all_features, axis=0).astype(np.float32)  # (N, 60)
    boards_arr = np.stack(all_boards, axis=0).astype(np.int8)          # (N, 10, 9)
    z_arr = np.array(all_z, dtype=np.float32)                          # (N,)
    move_scores_arr = np.array(all_move_scores, dtype=object)          # 列表的列表（变长）
    moves_order_arr = np.array(all_moves_order, dtype=object)          # 列表的列表（变长）

    np.savez(
        args.out,
        features=features_arr,
        boards=boards_arr,
        move_scores=move_scores_arr,
        z_values=z_arr,
        moves_order=moves_order_arr,
    )

    total_time = time.time() - t_start
    print(f"完成：共 {args.games} 局，{len(all_features)} 个局面，总耗时 {total_time/60:.1f} 分钟")
    print(f"数据已保存到 {args.out}")


if __name__ == "__main__":
    main()
