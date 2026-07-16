#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化测试：快速定位问题
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_game import ChessGame, RED, BLACK
from ai_engine import AIEngine


def test_single_move():
    """测试单步走棋"""
    print("=== 测试单步走棋 ===")
    
    # 创建游戏和引擎
    game = ChessGame()
    engine = AIEngine(disable_nn=True)  # 先禁用NN，简化问题
    
    print(f"当前回合: {'红方' if game.current_turn == RED else '黑方'}")
    
    # 获取红方走法
    moves = game.get_all_moves(RED)
    print(f"红方合法走法数: {len(moves)}")
    
    if moves:
        print("尝试走第一步...")
        start_time = time.time()
        
        # 使用低深度测试
        engine._score_to_ai_params = lambda x: (1, 0.0, 0.05, 0.0)  # 强制深度1，不放水
        
        move = engine.get_move(game, skill_score=100, color=RED)
        elapsed = time.time() - start_time
        
        print(f"AI选择走法: {move}")
        print(f"耗时: {elapsed:.2f}秒")
        
        if move:
            success = game.make_move(*move)
            print(f"走棋成功: {success}")
            if success:
                print("走棋成功！")
            else:
                print("走棋失败！")
        else:
            print("AI未返回走法")
    else:
        print("无合法走法")


def test_simple_game():
    """测试简化游戏（深度1，无NN）"""
    print("\n=== 测试简化游戏 ===")
    
    engine_red = AIEngine(disable_nn=True)
    engine_black = AIEngine(disable_nn=True)
    
    # 强制低深度
    engine_red._score_to_ai_params = lambda x: (1, 0.0, 0.05, 0.0)
    engine_black._score_to_ai_params = lambda x: (1, 0.0, 0.05, 0.0)
    
    game = ChessGame()
    move_count = 0
    
    while not game.game_over and move_count < 10:
        print(f"\n--- 第 {move_count + 1} 步 ---")
        
        if game.current_turn == RED:
            print("红方走棋...")
            start = time.time()
            move = engine_red.get_move(game, skill_score=100, color=RED)
            elapsed = time.time() - start
            print(f"红方走法: {move}, 耗时: {elapsed:.2f}秒")
        else:
            print("黑方走棋...")
            start = time.time()
            move = engine_black.get_move(game, skill_score=100, color=BLACK)
            elapsed = time.time() - start
            print(f"黑方走法: {move}, 耗时: {elapsed:.2f}秒")
        
        if move:
            success = game.make_move(*move)
            print(f"走棋成功: {success}")
            if success:
                move_count += 1
            else:
                print("走棋失败，游戏终止")
                break
        else:
            print("无走法，游戏终止")
            break
    
    print(f"\n游戏结束，总步数: {move_count}")


if __name__ == "__main__":
    test_single_move()
    test_simple_game()