#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版本：解决AI走法颜色错误问题
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_game import ChessGame, RED, BLACK, PIECE_CHARS
from ai_engine import AIEngine


class ExperimentConfig:
    def __init__(self, name, disable_nn=False, skill_score=80):
        self.name = name
        self.disable_nn = disable_nn
        self.skill_score = skill_score


class ExperimentResult:
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_nodes = 0
        self.total_moves = 0
    
    @property
    def win_rate(self):
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0
    
    @property
    def avg_nodes(self):
        return self.total_nodes / self.total_moves if self.total_moves > 0 else 0


def create_engine(config):
    return AIEngine(disable_nn=config.disable_nn)


def play_game(engine1, engine2, config1, config2, max_moves=200):
    """engine1控制红方，engine2控制黑方"""
    game = ChessGame()
    nodes1, nodes2 = 0, 0
    move_count = 0
    
    while not game.game_over and move_count < max_moves:
        if game.current_turn == RED:
            saved_nodes = engine1._nodes
            move = engine1.get_move(game, config1.skill_score, color=RED)
            nodes1 += engine1._nodes - saved_nodes
        else:
            saved_nodes = engine2._nodes
            move = engine2.get_move(game, config2.skill_score, color=BLACK)
            nodes2 += engine2._nodes - saved_nodes
        
        if move:
            success = game.make_move(*move)
            if success:
                move_count += 1
            else:
                # 走法无效，跳过
                print(f"无效走法: {move}")
        else:
            break
    
    return game.winner, nodes1, nodes2, move_count


def run_experiment(config1, config2, num_games=10):
    engine1 = create_engine(config1)
    engine2 = create_engine(config2)
    
    result = ExperimentResult()
    
    for i in range(num_games):
        if i % 5 == 0:
            print(f"\r    进度: {i}/{num_games}", end="")
            sys.stdout.flush()
        
        if i % 2 == 0:
            # 奇数局：config1=红方, config2=黑方
            winner, nodes1, nodes2, moves = play_game(engine1, engine2, config1, config2)
            if winner == RED:
                result.wins += 1
            elif winner == BLACK:
                result.losses += 1
            else:
                result.draws += 1
        else:
            # 偶数局：交换颜色
            winner, nodes2, nodes1, moves = play_game(engine2, engine1, config2, config1)
            if winner == BLACK:
                result.wins += 1
            elif winner == RED:
                result.losses += 1
            else:
                result.draws += 1
        
        result.total_nodes += nodes1 + nodes2
        result.total_moves += moves
    
    print(f"\r    进度: {num_games}/{num_games}")
    return result


def main():
    print("=== AI走棋逻辑对比实验（修复版）===")
    print("核心论点：AI走棋不仅是搜索，还有神经网络")
    print("="*50)
    
    configs = [
        ExperimentConfig(name="A组-纯搜索", disable_nn=True, skill_score=80),
        ExperimentConfig(name="B组-搜索+CNN", disable_nn=False, skill_score=80),
        ExperimentConfig(name="C组-搜索+价值网络", disable_nn=False, skill_score=80),
        ExperimentConfig(name="D组-完整方案", disable_nn=False, skill_score=80),
    ]
    
    results = {}
    
    for i, config1 in enumerate(configs):
        for j, config2 in enumerate(configs):
            if i >= j:
                continue
            
            print(f"\n【{config1.name}】 vs 【{config2.name}】")
            result = run_experiment(config1, config2, num_games=10)
            
            results[(config1.name, config2.name)] = {
                'win_rate': result.win_rate,
                'avg_nodes': result.avg_nodes,
                'wins': result.wins,
                'losses': result.losses,
                'draws': result.draws
            }
            
            print(f"  结果: {result.wins}胜 / {result.losses}负 / {result.draws}平")
            print(f"  胜率: {result.win_rate:.2%}")
            print(f"  平均搜索节点: {result.avg_nodes:.1f}")
    
    # 输出胜率矩阵
    print("\n" + "="*50)
    print("胜率矩阵（行方胜率）:")
    col_names = [c.name[:8] for c in configs]
    header = " " * 12 + " | ".join(f"{name:^12}" for name in col_names)
    print(header)
    print("-" * len(header))
    
    for i, config1 in enumerate(configs):
        row = f"{config1.name[:10]:<12}"
        for j, config2 in enumerate(configs):
            if i == j:
                row += " |    -----"
            elif (config1.name, config2.name) in results:
                wr = results[(config1.name, config2.name)]['win_rate']
                row += f" |   {wr:6.1%}"
            else:
                wr = 1.0 - results[(config2.name, config1.name)]['win_rate']
                row += f" |   {wr:6.1%}"
        print(row)


if __name__ == "__main__":
    main()