#!/usr/bin/env python3
"""一键 demo：训练 + 评估 + 可视化，三个问题全部跑一遍。

用法:
    python scripts/demo.py --episodes 250 --out outputs

产出（outputs/ 下）:
    {tsp,knapsack,flowshop}_policy.pkl      训练好的策略
    {problem}_train_curve.png               训练曲线
    {problem}_comparison.png                各算法对比条形图
    {problem}_spec_usage.png                自动设计算法的构件使用统计
    tsp_routes.png                          TSP 路径对比
    results.txt                             评估结果总表
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from auto_alg_design import FlowShop, Knapsack, TSP
from auto_alg_design.baselines.genetic_algorithm import GeneticAlgorithm
from auto_alg_design.baselines.simulated_annealing import SimulatedAnnealing
from auto_alg_design.components.runner import run_algorithm
from auto_alg_design.components.search_ops import _two_opt
from auto_alg_design.evaluate import EvalConfig, evaluate, format_specs, format_table
from auto_alg_design.rl.features import estimate_feature_stats
from auto_alg_design.rl.env import DesignEnv
from auto_alg_design.rl.trainer import TrainConfig, generate_algorithm, train
from auto_alg_design.visualize import (
    plot_comparison,
    plot_spec_usage,
    plot_training_curve,
    plot_tsp_tour,
)

PROBLEM_FACTORIES = {
    "tsp": lambda seed: TSP(n=30, seed=seed),
    "knapsack": lambda seed: Knapsack(n=50, seed=seed),
    "flowshop": lambda seed: FlowShop(n_jobs=15, n_machines=6, seed=seed),
}


def run_problem(name: str, episodes: int, out: str) -> str:
    problem = PROBLEM_FACTORIES[name](seed=0)
    cfg = TrainConfig(episodes=episodes, seed=0)
    print(f"\n{'=' * 60}\n[1/3] TRAIN  problem={name}  episodes={episodes}\n{'=' * 60}")
    result = train(problem, cfg, iterations=60)

    policy_path = os.path.join(out, f"{name}_policy.pkl")
    with open(policy_path, "wb") as f:
        pickle.dump(result.policy, f)
    plot_training_curve(result.rewards, result.eval_steps, result.eval_means,
                        os.path.join(out, f"{name}_train_curve.png"))

    print(f"\n{'=' * 60}\n[2/3] EVALUATE  problem={name}  (unseen instances)\n{'=' * 60}")
    eval_cfg = EvalConfig(n_test_instances=5, n_runs=4, seed=1)
    res = evaluate(problem, result.policy, eval_cfg)
    table = format_table(res)
    print(table)

    plot_comparison(res, os.path.join(out, f"{name}_comparison.png"))
    specs_text = format_specs(res)
    print(specs_text)
    plot_spec_usage(
        [s for m in res.methods if m.name == "rl_designed" for s in m.specs],
        os.path.join(out, f"{name}_spec_usage.png"),
    )

    # TSP 额外：各算法路径图
    if name == "tsp":
        _plot_tsp_routes(problem, result.policy, out)

    return table


def _plot_tsp_routes(problem: TSP, policy, out: str):
    rng = np.random.default_rng(7)
    inst = problem.random_instance(rng)
    feats_mean, feats_std = estimate_feature_stats(problem, seed=0)
    env = DesignEnv(problem=problem, instance=inst, feats_mean=feats_mean, feats_std=feats_std)

    # RL 自动设计算法
    rl_spec = generate_algorithm(policy, env, np.random.default_rng(11))
    rl_cost, rl_traj, rl_sol = run_algorithm(problem, inst, rl_spec,
                                             np.random.default_rng(11), record_trajectory=True)

    ga = GeneticAlgorithm(population=24, generations=40, seed=3)
    ga_sol, ga_cost = ga.solve(problem, inst)
    sa = SimulatedAnnealing(iterations=600, seed=5)
    sa_sol, sa_cost = sa.solve(problem, inst)

    # 参考：多起点 2-opt
    best_ref, best_c = None, float("inf")
    rng2 = np.random.default_rng(9)
    for _ in range(20):
        cand = problem.random_solution(inst, rng2)
        cand = _two_opt(problem, inst, cand, rng2, 2)
        c = problem.cost(inst, cand)
        if c < best_c:
            best_c, best_ref = c, cand

    plot_tsp_tour(
        inst,
        {"RL-designed": rl_sol, "GA": ga_sol, "SA": sa_sol, "2-opt ref": best_ref},
        os.path.join(out, "tsp_routes.png"),
    )
    print(f"[viz] TSP 实例: RL={rl_cost:.3f}  GA={ga_cost:.3f}  SA={sa_cost:.3f}  2-opt参考={best_c:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = os.path.join(args.out)
    os.makedirs(out, exist_ok=True)

    all_lines = ["=" * 70, "智能优化算法自动设计 - 方向1验证原型 实验结果", "=" * 70]
    for name in ["tsp", "knapsack", "flowshop"]:
        table = run_problem(name, args.episodes, out)
        all_lines.append(table)

    results_path = os.path.join(out, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"\n[demo] 全部完成，结果汇总 -> {results_path}")


if __name__ == "__main__":
    main()
