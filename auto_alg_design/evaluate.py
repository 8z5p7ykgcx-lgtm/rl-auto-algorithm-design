"""评估器：在「训练中未见过」的测试实例上对比各算法。

对比对象：
- RL-Designed：训练好的策略为每个实例自动生成的算法（多次重复取均值/最优）
- GA / SA / Random：人工设计与随机基线，同一实例同一预算量级

指标：平均成本、相对随机基线的改进率、最优成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baselines.genetic_algorithm import GeneticAlgorithm
from .baselines.simulated_annealing import SimulatedAnnealing
from .components.registry import describe
from .components.runner import run_algorithm
from .problems.base import Problem
from .rl.env import DesignEnv
from .rl.features import estimate_feature_stats
from .rl.policy import SoftmaxPolicy
from .rl.trainer import generate_algorithm


@dataclass
class EvalConfig:
    n_test_instances: int = 6
    n_runs: int = 5                  # 每个实例每个方法重复次数
    iterations: int = 60             # 自动设计算法的迭代预算
    n_search_steps: int = 4
    seed: int = 1
    ga_population: int = 24
    ga_generations: int = 40
    sa_iterations: int = 600         # 与自动算法迭代量级相当（含邻域评估）
    record_specs: bool = True


@dataclass
class MethodResult:
    name: str
    mean_cost: float
    best_cost: float
    std_cost: float
    mean_improve: float              # 相对随机基线的平均改进率
    specs: list[str] = field(default_factory=list)   # RL 生成的算法描述


@dataclass
class EvalResult:
    problem_name: str
    methods: list[MethodResult]
    random_mean: float
    n_instances: int


def evaluate(
    problem: Problem,
    policy: SoftmaxPolicy | None,
    cfg: EvalConfig | None = None,
) -> EvalResult:
    cfg = cfg or EvalConfig()
    rng = np.random.default_rng(cfg.seed)
    feats_mean, feats_std = estimate_feature_stats(problem, seed=cfg.seed)
    env = DesignEnv(
        problem=problem,
        instance=problem.random_instance(rng),
        feats_mean=feats_mean,
        feats_std=feats_std,
        n_search_steps=cfg.n_search_steps,
        iterations=cfg.iterations,
    )

    results: dict[str, dict] = {k: {"costs": [], "best": [], "impr": [], "specs": []}
                                for k in ["rl_designed", "ga", "sa", "random"]}
    random_total = 0.0
    n_count = 0

    for _ in range(cfg.n_test_instances):
        inst = problem.random_instance(rng)
        env.instance = inst

        # 随机基线（每个实例同预算：随机解均值）
        base_costs = [
            problem.cost(inst, problem.random_solution(inst, rng))
            for _ in range(20)
        ]
        base = float(np.mean(base_costs))
        random_total += base
        n_count += 1

        # RL 自动设计算法
        for _ in range(cfg.n_runs):
            spec = generate_algorithm(policy, env, rng)
            cost, _, _ = run_algorithm(problem, inst, spec, rng)
            results["rl_designed"]["costs"].append(cost)
            results["rl_designed"]["impr"].append((base - cost) / abs(base))
            if cfg.record_specs:
                results["rl_designed"]["specs"].append(spec.describe())

        # GA
        for _ in range(cfg.n_runs):
            ga = GeneticAlgorithm(
                population=cfg.ga_population,
                generations=cfg.ga_generations,
                seed=int(rng.integers(0, 10000)),
            )
            _, cost = ga.solve(problem, inst)
            results["ga"]["costs"].append(cost)
            results["ga"]["impr"].append((base - cost) / abs(base))

        # SA
        for _ in range(cfg.n_runs):
            sa = SimulatedAnnealing(
                iterations=cfg.sa_iterations,
                seed=int(rng.integers(0, 10000)),
            )
            _, cost = sa.solve(problem, inst)
            results["sa"]["costs"].append(cost)
            results["sa"]["impr"].append((base - cost) / abs(base))

        # Random（单次随机解均值即可）
        results["random"]["costs"].append(base)
        results["random"]["impr"].append(0.0)

    methods = []
    for name in ["rl_designed", "ga", "sa", "random"]:
        costs = np.asarray(results[name]["costs"])
        impr = np.asarray(results[name]["impr"])
        methods.append(
            MethodResult(
                name=name,
                mean_cost=float(costs.mean()),
                best_cost=float(costs.min()),
                std_cost=float(costs.std()),
                mean_improve=float(impr.mean()),
                specs=results[name]["specs"],
            )
        )

    return EvalResult(
        problem_name=problem.name,
        methods=methods,
        random_mean=random_total / max(n_count, 1),
        n_instances=cfg.n_test_instances,
    )


def format_table(res: EvalResult) -> str:
    """把评估结果格式化为可打印表格。"""
    lines = [f"\n===== 评估结果：{res.problem_name}  (测试实例数={res.n_instances}) ====="]
    header = f"{'方法':<14}{'平均成本':>14}{'最优成本':>14}{'平均改进率':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for m in res.methods:
        lines.append(
            f"{m.name:<14}{m.mean_cost:>14.4f}{m.best_cost:>14.4f}{m.mean_improve:>+11.2%}"
        )
    return "\n".join(lines)


def format_specs(res: EvalResult) -> str:
    """展示 RL 自动设计出的算法（去重后）。"""
    seen = []
    for m in res.methods:
        if m.name == "rl_designed":
            for s in m.specs:
                if s not in seen:
                    seen.append(s)
    lines = [f"--- RL 自动设计出的算法（去重 {len(seen)} 种）---"]
    lines.extend(seen[:12])
    return "\n".join(lines)
