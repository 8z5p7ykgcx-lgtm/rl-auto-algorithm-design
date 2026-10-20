"""算法执行器：把自动生成的 AlgorithmSpec 跑在某个问题实例上。

这是「自动设计的算法」真正被求解的部分：按 init → 多步搜索(带参数) → 接受规则
的顺序执行，记录迭代轨迹（cost 序列）供可视化与奖励计算。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..problems.base import Problem
from .registry import get


def run_algorithm(
    problem: Problem,
    instance: dict[str, Any],
    spec: "AlgorithmSpec",
    rng: np.random.Generator,
    record_trajectory: bool = False,
) -> tuple[float, np.ndarray | None, np.ndarray]:
    """运行自动设计的算法，返回 (最终成本, 轨迹或None, 最优解)。"""
    init = get(spec.init_id)
    accept = get(spec.accept_id)

    # 初始化（init 构件同样带参数档位，由 RL 一并决策）
    sol = np.asarray(init.fn(problem, instance, rng, spec.init_param))

    memory: Any = None
    best_sol = sol.copy()
    traj: list[float] = []
    cur = sol

    for step in range(spec.iterations):
        # 每步：选一个搜索构件执行（自动组合出的搜索序列循环使用）
        cid, param = spec.search_steps[step % len(spec.search_steps)]
        comp = get(cid)
        candidate = comp.fn(problem, instance, cur, rng, param)
        candidate = problem.repair(instance, candidate)

        # 注入进度供 sa_accept 降温
        if isinstance(memory, dict):
            memory["progress"] = step / max(spec.iterations - 1, 1)
        cur, memory = accept.fn(problem, instance, cur, candidate, rng, spec.accept_param, memory=memory)

        if problem.cost(instance, cur) < problem.cost(instance, best_sol):
            best_sol = cur.copy()
        if record_trajectory:
            traj.append(problem.cost(instance, best_sol))

    final = best_sol
    cost = problem.cost(instance, final)
    return cost, (np.asarray(traj) if record_trajectory else None), final
