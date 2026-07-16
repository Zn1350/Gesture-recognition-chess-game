#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows兼容的调试版本
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_game import ChessGame, RED, BLACK


class TimeoutResult:
    """超时结果封装"""
    def __init__(self):
        self.result = None
        self.timed_out = False
        self.error = None


def run_with_timeout(func, timeout=5):
    """带超时的函数执行"""
    result = TimeoutResult()
    
    def target():
        try:
            result.result = func()
        except Exception as e:
            result.error = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        result.timed_out = True
    return result


def test_quiescence():
    """测试静态搜索"""
    print("\n=== 测试静态搜索 ===")
    
    from ai_engine import AIEngine
    
    game = ChessGame()
    engine = AIEngine(disable_nn=True)
    
    def run_search():
        return engine._quiescence(game, -999999, 999999, RED)
    
    print("测试静态搜索...")
    start = time.time()
    result = run_with_timeout(run_search, timeout=10)
    elapsed = time.time() - start
    
    if result.timed_out:
        print(f"ERROR: 静态搜索超时！({elapsed:.2f}秒)")
    elif result.error:
        print(f"ERROR: {result.error}")
    else:
        print(f"静态搜索评分: {result.result}, 耗时: {elapsed:.2f}秒")


def test_minimax_depth1():
    """测试深度1的minimax搜索"""
    print("\n=== 测试深度1 Minimax ===")
    
    from ai_engine import AIEngine
    
    game = ChessGame()
    engine = AIEngine(disable_nn=True)
    
    def run_search():
        moves = game.get_all_moves(RED)
        if not moves:
            return None
        g = game.board_copy()
        g.make_move(*moves[0])
        return engine._minimax(g, 0, -999999, 999999, False)
    
    print("测试深度0搜索...")
    start = time.time()
    result = run_with_timeout(run_search, timeout=10)
    elapsed = time.time() - start
    
    if result.timed_out:
        print(f"ERROR: 深度0搜索超时！({elapsed:.2f}秒)")
    elif result.error:
        print(f"ERROR: {result.error}")
    else:
        print(f"深度0搜索评分: {result.result}, 耗时: {elapsed:.2f}秒")


def test_minimax_full():
    """测试完整的minimax搜索"""
    print("\n=== 测试完整Minimax搜索 ===")
    
    from ai_engine import AIEngine
    
    game = ChessGame()
    engine = AIEngine(disable_nn=True)
    engine._tt = {}  # 初始化置换表
    engine._history = {}  # 初始化历史启发
    
    def run_search():
        moves = game.get_all_moves(RED)
        if not moves:
            return None
        g = game.board_copy()
        g.make_move(*moves[0])
        return engine._minimax(g, 1, -999999, 999999, False)
    
    print("测试深度1搜索...")
    start = time.time()
    result = run_with_timeout(run_search, timeout=10)
    elapsed = time.time() - start
    
    if result.timed_out:
        print(f"ERROR: 深度1搜索超时！({elapsed:.2f}秒)")
    elif result.error:
        print(f"ERROR: {result.error}")
    else:
        print(f"深度1搜索评分: {result.result}, 耗时: {elapsed:.2f}秒")
        print(f"搜索节点数: {engine._nodes}")


def test_best_move():
    """测试最佳走法选择"""
    print("\n=== 测试最佳走法选择 ===")
    
    from ai_engine import AIEngine
    
    game = ChessGame()
    engine = AIEngine(disable_nn=True)
    
    def run_search():
        moves = game.get_all_moves(RED)
        return engine._best_move(game, moves, depth=1, skill_score=100)
    
    print("测试_best_move...")
    start = time.time()
    result = run_with_timeout(run_search, timeout=10)
    elapsed = time.time() - start
    
    if result.timed_out:
        print(f"ERROR: _best_move超时！({elapsed:.2f}秒)")
    elif result.error:
        print(f"ERROR: {result.error}")
    else:
        print(f"最佳走法: {result.result}, 耗时: {elapsed:.2f}秒")


def test_get_move():
    """测试完整的get_move"""
    print("\n=== 测试完整get_move ===")
    
    from ai_engine import AIEngine
    
    game = ChessGame()
    engine = AIEngine(disable_nn=True)
    
    def run_search():
        return engine.get_move(game, skill_score=100, color=RED)
    
    print("测试get_move...")
    start = time.time()
    result = run_with_timeout(run_search, timeout=10)
    elapsed = time.time() - start
    
    if result.timed_out:
        print(f"ERROR: get_move超时！({elapsed:.2f}秒)")
    elif result.error:
        print(f"ERROR: {result.error}")
    else:
        print(f"走法: {result.result}, 耗时: {elapsed:.2f}秒")


if __name__ == "__main__":
    test_quiescence()
    test_minimax_depth1()
    test_minimax_full()
    test_best_move()
    test_get_move()