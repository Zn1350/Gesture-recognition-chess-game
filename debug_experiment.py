#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试版本：简化实验逻辑，添加详细日志
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_game import ChessGame, RED, BLACK, PIECE_CHARS
from ai_engine import AIEngine


def print_board(game):
    """打印棋盘状态"""
    print("\n当前棋盘:")
    print("  0 1 2 3 4 5 6 7 8")
    for r in range(10):
        row = f"{r} "
        for c in range(9):
            piece = game.board[r][c]
            if piece == 0:
                row += ". "
            else:
                row += f"{PIECE_CHARS.get(piece, '?')} "
        print(row)
    print(f"当前回合: {'红方' if game.current_turn == RED else '黑方'}")


def test_single_game():
    """测试单局对弈，添加详细日志"""
    print("=== 测试单局对弈 ===")
    
    # 创建引擎
    print("\n创建AI引擎...")
    engine1 = AIEngine(disable_nn=False)
    engine2 = AIEngine(disable_nn=False)
    
    print(f"Engine1 NN可用: {engine1.nn_available}")
    print(f"Engine1 模型: {engine1.nn_evaluator.model}")
    print(f"Engine2 NN可用: {engine2.nn_available}")
    
    # 初始化游戏
    game = ChessGame()
    print_board(game)
    
    move_count = 0
    max_moves = 30
    
    while not game.game_over and move_count < max_moves:
        print(f"\n--- 第 {move_count + 1} 步 ---")
        
        if game.current_turn == RED:
            print("红方走棋...")
            move = engine1.get_move(game, skill_score=100)
        else:
            print("黑方走棋...")
            move = engine2.get_move(game, skill_score=100)
        
        print(f"走法: {move}")
        
        if move:
            fr, fc, tr, tc = move
            piece = game.board[fr][fc]
            captured = game.board[tr][tc]
            print(f"移动: {PIECE_CHARS.get(piece, '?')} ({fr},{fc}) -> ({tr},{tc})")
            if captured != 0:
                print(f"吃掉: {PIECE_CHARS.get(captured, '?')}")
            
            success = game.make_move(*move)
            print(f"走棋成功: {success}")
        else:
            print("无子可走!")
            break
        
        print_board(game)
        move_count += 1
    
    print(f"\n=== 游戏结束 ===")
    print(f"总步数: {move_count}")
    print(f"游戏结束: {game.game_over}")
    print(f"胜者: {'红方' if game.winner == RED else '黑方' if game.winner == BLACK else '平局'}")


def test_ai_move_generation():
    """测试AI走法生成"""
    print("\n=== 测试AI走法生成 ===")
    
    game = ChessGame()
    engine = AIEngine(disable_nn=False)
    
    # 获取所有合法走法
    moves = game.get_all_moves(RED)
    print(f"红方合法走法数: {len(moves)}")
    
    if moves:
        print("前5个走法:")
        for m in moves[:5]:
            fr, fc, tr, tc = m
            piece = game.board[fr][fc]
            print(f"  {PIECE_CHARS.get(piece, '?')} ({fr},{fc}) -> ({tr},{tc})")
    
    # 测试AI选择走法
    print("\nAI选择走法...")
    move = engine.get_move(game, skill_score=100)
    print(f"AI选择: {move}")


if __name__ == "__main__":
    test_single_game()
    test_ai_move_generation()