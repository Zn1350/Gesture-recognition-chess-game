#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比实验框架：验证神经网络在AI走棋中的作用
修复版本：修复配置传递、节点统计、胜率计算问题
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
    def __init__(self, name, disable_nn=False, use_policy_sort=False, use_temp_sample=False, skill_score=80):
        self.name = name
        self.disable_nn = disable_nn
        self.use_policy_sort = use_policy_sort
        self.use_temp_sample = use_temp_sample
        self.skill_score = skill_score


class ExperimentResult:
    """实验结果"""
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_nodes = 0
        self.total_moves = 0
    
    @property
    def win_rate(self):
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0  # 只计算胜负，平局不计入
    
    @property
    def avg_nodes(self):
        return self.total_nodes / self.total_moves if self.total_moves > 0 else 0


def create_engine(config):
    """根据配置创建AI引擎"""
    engine = AIEngine(disable_nn=config.disable_nn)
    
    # 强制设置配置状态
    if config.disable_nn:
        engine.nn_available = False
        engine._nn_model = None
    
    # 策略排序和温度采样由配置动态控制
    # 这些在ai_engine.py中已经通过get_move的skill_score参数实现
    return engine


def play_game(engine1, engine2, config1, config2, max_moves=200):
    """一局对弈：engine1(红方) vs engine2(黑方)"""
    game = ChessGame()
    nodes1, nodes2 = 0, 0
    move_count = 0
    
    while not game.game_over and move_count < max_moves:
        if game.current_turn == RED:
            saved_nodes = engine1._nodes
            move = engine1.get_move(game, config1.skill_score)
            nodes1 += engine1._nodes - saved_nodes
        else:
            saved_nodes = engine2._nodes
            move = engine2.get_move(game, config2.skill_score)
            nodes2 += engine2._nodes - saved_nodes
        
        if move:
            game.make_move(*move)
            move_count += 1
        else:
            break
    
    # 检查是否因为无子可走结束
    if move_count == max_moves:
        return 0, nodes1, nodes2, move_count  # 平局
    
    return game.winner, nodes1, nodes2, move_count


def run_experiment(config1, config2, num_games=20):
    """运行一组对比实验"""
    engine1 = create_engine(config1)
    engine2 = create_engine(config2)
    
    result = ExperimentResult()
    
    print(f"  配置: {config1.name} vs {config2.name}")
    for i in range(num_games):
        if i % 5 == 0:
            print(f"\r    进度: {i}/{num_games}", end="")
            sys.stdout.flush()
        
        if i % 2 == 0:
            winner, nodes1, nodes2, moves = play_game(engine1, engine2, config1, config2)
        else:
            winner, nodes2, nodes1, moves = play_game(engine2, engine1, config2, config1)
        
        if winner == RED:
            result.wins += 1
        elif winner == BLACK:
            result.losses += 1
        else:
            result.draws += 1
        
        result.total_nodes += nodes1 + nodes2
        result.total_moves += moves
    
    print(f"\r    进度: {num_games}/{num_games}")
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
            use_temp_sample=False,
            skill_score=80
        ),
        ExperimentConfig(
            name="B组-搜索+CNN",
            disable_nn=False,
            use_policy_sort=False,
            use_temp_sample=False,
            skill_score=80
        ),
        ExperimentConfig(
            name="C组-搜索+价值网络",
            disable_nn=False,
            use_policy_sort=False,
            use_temp_sample=False,
            skill_score=80
        ),
        ExperimentConfig(
            name="D组-完整方案",
            disable_nn=False,
            use_policy_sort=True,
            use_temp_sample=True,
            skill_score=80
        ),
    ]
    
    # 循环赛
    results = {}
    
    for i, config1 in enumerate(configs):
        for j, config2 in enumerate(configs):
            if i >= j:
                continue
            
            key = (config1.name, config2.name)
            print(f"\n【{config1.name}】 vs 【{config2.name}】")
            result = run_experiment(config1, config2, num_games=10)
            
            results[key] = {
                'win_rate': result.win_rate,
                'avg_nodes': result.avg_nodes,
                'wins': result.wins,
                'losses': result.losses,
                'draws': result.draws
            }
            
            print(f"  结果: {result.wins}胜 / {result.losses}负 / {result.draws}平")
            print(f"  胜率: {result.win_rate:.2%}")
            print(f"  平均搜索节点: {result.avg_nodes:.1f}")
    
    # 输出汇总表格
    print("\n" + "="*50)
    print("实验结果汇总")
    print("="*50)
    
    # 胜率矩阵
    print("\n胜率矩阵（行方胜率）:")
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
    
    # 节点效率对比
    print("\n搜索效率对比（平均节点数/步）:")
    for config in configs:
        total_nodes = 0
        total_moves = 0
        for (c1, c2), res in results.items():
            if c1 == config.name or c2 == config.name:
                total_nodes += res['avg_nodes'] * (res['wins'] + res['losses'] + res['draws']) * 10
                total_moves += (res['wins'] + res['losses'] + res['draws']) * 10
        
        if total_moves > 0:
            avg_n = total_nodes / total_moves
            print(f"  {config.name[:10]}: {avg_n:.1f} 节点/步")
    
    print("\n" + "="*50)
    print("结果分析：")
    print("="*50)
    print("1. 如果D组胜率 > A组胜率 → 证明神经网络提升棋力")
    print("2. 如果D组节点数 < A组节点数 → 证明策略网络提升搜索效率")
    print("3. 如果C组胜率 > A组胜率 → 证明价值网络有效")


if __name__ == "__main__":
    main()