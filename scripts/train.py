#!/usr/bin/env python3
"""训练入口：为指定问题类型训练自动算法设计策略。

用法:
    python scripts/train.py --problem tsp --episodes 300 --iterations 60 --out outputs

产出:
    outputs/<problem>_policy.pkl        训练好的策略
    outputs/<problem>_train_curve.png   训练曲线
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_alg_design import FlowShop, Knapsack, TSP
from auto_alg_design.rl.trainer import TrainConfig, train
from auto_alg_design.visualize import plot_training_curve

PROBLEMS = {"tsp": TSP, "knapsack": Knapsack, "flowshop": FlowShop}


def make_problem(name: str, seed: int):
    if name == "tsp":
        return TSP(n=30, seed=seed)
    if name == "knapsack":
        return Knapsack(n=50, seed=seed)
    return FlowShop(n_jobs=15, n_machines=6, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(PROBLEMS), default="tsp")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    problem = make_problem(args.problem, seed=args.seed)
    cfg = TrainConfig(episodes=args.episodes, lr=args.lr, seed=args.seed)

    print(f"[train] problem={args.problem} episodes={args.episodes} iterations={args.iterations}")
    result = train(problem, cfg, iterations=args.iterations)

    # 保存策略
    policy_path = os.path.join(args.out, f"{args.problem}_policy.pkl")
    with open(policy_path, "wb") as f:
        pickle.dump(result.policy, f)
    print(f"[train] policy saved -> {policy_path}")

    # 训练曲线
    curve_path = os.path.join(args.out, f"{args.problem}_train_curve.png")
    plot_training_curve(result.rewards, result.eval_steps, result.eval_means, curve_path)

    print(f"[train] best designed algorithm: {result.best_spec_text}  (reward={result.best_reward:+.4f})")


if __name__ == "__main__":
    main()
