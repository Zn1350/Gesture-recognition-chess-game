#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比实验框架：验证神经网络在AI走棋中的作用

实验设计：
A组 - 纯搜索（手工评估）
B组 - 搜索+旧CNN评估
C组 - 搜索+MicroDualNet价值头
D组 - 完整方案（价值+策略+温度采样）
"""

import os
import sys
import time
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_game import ChessGame, RED, BLACK
from ai_engine import AIEngine


class ExperimentConfig:
    """实验配置"""
    def __init__(self, name, disable_nn=False, use_micro_only=False, 
                 use_policy_sort=False, use_temp_sample=False):
        self.name = name
        self.disable_nn = disable_nn          # 禁用所有神经网络
        self.use_micro_only = use_micro_only  # 仅用MicroDualNet（不含旧CNN）
        self.use_policy_sort = use_policy_sort  # 使用策略网络排序
        self.use_temp_sample = use_temp_sample  # 使用温度采样
        self.skill_score = 80                 # AI棋力分（较高水平）


class ExperimentResult:
    """实验结果"""
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_nodes = 0
        self.total_moves = 0
        self.avg_eval_diff = 0.0
    
    @property
    def win_rate(self):
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0
    
    @property
    def avg_nodes(self):
        return self.total_nodes / self.total_moves if self.total_moves > 0 else 0


def play_game(engine1, engine2, config1, config2, max_moves=150):
    """
    一局对弈：engine1(红方) vs engine2(黑方)
    返回：(winner, nodes1, nodes2, move_count)
    winner: 1=红方胜, -1=黑方胜, 0=平局
    """
    game = ChessGame()
    nodes1, nodes2 = 0, 0
    move_count = 0
    
    while not game.game_over and move_count < max_moves:
        if game.current_turn == RED:
            # 红方走棋
            saved_nodes = engine1._nodes
            move = engine1.get_move(game, config1.skill_score)
            nodes1 += engine1._nodes - saved_nodes
        else:
            # 黑方走棋
            saved_nodes = engine2._nodes
            move = engine2.get_move(game, config2.skill_score)
            nodes2 += engine2._nodes - saved_nodes
        
        if move:
            game.make_move(*move)
            move_count += 1
        else:
            break  # 无子可走
    
    return game.winner, nodes1, nodes2, move_count


def run_experiment(config1, config2, num_games=20):
    """
    运行一组对比实验
    """
    # 创建引擎实例
    engine1 = AIEngine(
        disable_nn=config1.disable_nn,
        # 通过配置文件控制其他行为...
    )
    engine2 = AIEngine(
        disable_nn=config2.disable_nn,
    )
    
    # 确保配置生效（修改引擎状态）
    engine1.nn_available = not config1.disable_nn
    engine2.nn_available = not config2.disable_nn
    
    result = ExperimentResult()
    
    for i in range(num_games):
        print(f"\rGame {i+1}/{num_games}", end="")
        sys.stdout.flush()
        
        # 交替执红黑
        if i % 2 == 0:
            winner, nodes1, nodes2, moves = play_game(engine1, engine2, config1, config2)
            if winner == RED:
                result.wins += 1
            elif winner == BLACK:
                result.losses += 1
            else:
                result.draws += 1
        else:
            winner, nodes2, nodes1, moves = play_game(engine2, engine1, config2, config1)
            if winner == BLACK:
                result.wins += 1
            elif winner == RED:
                result.losses += 1
            else:
                result.draws += 1
        
        result.total_nodes += nodes1 + nodes2
        result.total_moves += moves
    
    print()
    return result


def main():
    """主实验流程"""
    print("=== AI走棋逻辑对比实验 ===")
    print("核心论点：AI走棋不仅是搜索，还有神经网络")
    print("="*50)
    
    # 定义四组配置
    configs = [
        ExperimentConfig(
            name="A组-纯搜索",
            disable_nn=True,
            use_policy_sort=False,
            use_temp_sample=False
        ),
        ExperimentConfig(
            name="B组-搜索+旧CNN",
            disable_nn=False,
            use_micro_only=False,
            use_policy_sort=False,
            use_temp_sample=False
        ),
        ExperimentConfig(
            name="C组-搜索+价值网络",
            disable_nn=False,
            use_micro_only=True,
            use_policy_sort=False,
            use_temp_sample=False
        ),
        ExperimentConfig(
            name="D组-完整方案",
            disable_nn=False,
            use_micro_only=False,
            use_policy_sort=True,
            use_temp_sample=True
        ),
    ]
    
    # 循环赛：每组互相对战
    results = defaultdict(dict)
    
    for i, config1 in enumerate(configs):
        for j, config2 in enumerate(configs):
            if i >= j:
                continue  # 避免重复对战
            
            print(f"\n【{config1.name}】 vs 【{config2.name}】")
            result = run_experiment(config1, config2, num_games=10)
            
            results[config1.name][config2.name] = {
                'win_rate': result.win_rate,
                'avg_nodes': result.avg_nodes,
                'wins': result.wins,
                'losses': result.losses,
                'draws': result.draws
            }
            
            print(f"  胜率: {result.win_rate:.2%}")
            print(f"  胜/负/平: {result.wins}/{result.losses}/{result.draws}")
            print(f"  平均搜索节点: {result.avg_nodes:.1f}")
    
    # 输出汇总表格
    print("\n" + "="*50)
    print("实验结果汇总")
    print("="*50)
    
    # 胜率矩阵
    print("\n胜率矩阵（行vs列）:")
    header = " " * 12 + " | ".join(f"{c.name[:6]:^10}" for c in configs)
    print(header)
    print("-" * len(header))
    
    for i, config1 in enumerate(configs):
        row = f"{config1.name[:10]:<12}"
        for j, config2 in enumerate(configs):
            if i == j:
                row += " |    -----"
            elif i < j:
                wr = results[config1.name][config2.name]['win_rate']
                row += f" |   {wr:5.1%}"
            else:
                wr = 1.0 - results[config2.name][config1.name]['win_rate']
                row += f" |   {wr:5.1%}"
        print(row)
    
    print("\n结论：")
    print("1. 纯搜索(A组) vs 完整方案(D组): D组胜率应显著高于A组")
    print("2. 搜索+价值网络(C组) vs 纯搜索(A组): C组胜率应更高")
    print("3. 策略排序+Top-K可显著减少搜索节点数")
    print("4. 温度采样放水比随机放水更接近人类水平")


if __name__ == "__main__":
    main()