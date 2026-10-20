#!/usr/bin/env python3
"""评估入口：在未见实例上对比 RL 自动设计算法 vs GA/SA/Random。

用法:
    python scripts/evaluate.py --problem tsp --policy outputs/tsp_policy.pkl --out outputs
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_alg_design import FlowShop, Knapsack, TSP
from auto_alg_design.evaluate import EvalConfig, evaluate, format_specs, format_table
from auto_alg_design.visualize import plot_comparison, plot_spec_usage


def make_problem(name: str, seed: int):
    if name == "tsp":
        return TSP(n=30, seed=seed)
    if name == "knapsack":
        return Knapsack(n=50, seed=seed)
    return FlowShop(n_jobs=15, n_machines=6, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["tsp", "knapsack", "flowshop"], default="tsp")
    ap.add_argument("--policy", default=None, help="训练好的策略 pkl 路径（缺省则随机策略对照）")
    ap.add_argument("--n-instances", type=int, default=6)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    problem = make_problem(args.problem, seed=args.seed)

    policy = None
    if args.policy and os.path.exists(args.policy):
        with open(args.policy, "rb") as f:
            policy = pickle.load(f)
        print(f"[eval] loaded policy from {args.policy}")

    cfg = EvalConfig(
        n_test_instances=args.n_instances,
        n_runs=args.n_runs,
        iterations=args.iterations,
        seed=args.seed,
    )
    res = evaluate(problem, policy, cfg)

    print(format_table(res))
    if policy is not None:
        print(format_specs(res))
        plot_spec_usage([s for m in res.methods if m.name == "rl_designed" for s in m.specs],
                        os.path.join(args.out, f"{args.problem}_spec_usage.png"))
    plot_comparison(res, os.path.join(args.out, f"{args.problem}_comparison.png"))


if __name__ == "__main__":
    main()
