"""初始化构件：生成初始解。

- random_init       随机初始化（适用于所有编码）
- greedy_tsp        TSP 最近邻贪心
- greedy_kp         背包按价值密度贪心（带修复）
- neh_init          流水车间 NEH 启发式
"""

from __future__ import annotations

import numpy as np

from ..problems.base import Problem
from ..problems.tsp import TSP
from ..problems.knapsack import Knapsack
from ..problems.flowshop import FlowShop
from .registry import Component, register


def _random_init(problem, instance, rng, param) -> np.ndarray:
    return problem.random_solution(instance, rng)


def _greedy_tsp(problem, instance, rng, param) -> np.ndarray:
    # param=0 用最近邻，param=1 用随机起点最近邻（同一构件、不同档位）
    return TSP.nearest_neighbor_tour(instance, rng)


def _greedy_kp(problem, instance, rng, param) -> np.ndarray:
    values, weights, capacity = instance["values"], instance["weights"], instance["capacity"]
    density = values / weights
    # param=0 完全贪心；param=1 加入 10% 随机扰动（探索性初始化）
    if param == 1:
        density = density * (1.0 + 0.2 * rng.uniform(-1, 1, size=len(density)))
    order = np.argsort(-density)
    sol = np.zeros(len(values), dtype=int)
    total_w = 0.0
    for i in order:
        if total_w + weights[i] <= capacity:
            sol[i] = 1
            total_w += weights[i]
    return sol


def _neh(problem, instance, rng, param) -> np.ndarray:
    return FlowShop.neh_init(instance)


register(Component("random_init", "init", "any", [0], _random_init, "随机初始化"))
register(Component("greedy_tsp", "init", "permutation", [0, 1], _greedy_tsp, "最近邻贪心初始化", problems={"tsp"}))
register(Component("greedy_kp", "init", "binary", [0, 1], _greedy_kp, "价值密度贪心初始化", problems={"knapsack"}))
register(Component("neh_init", "init", "permutation", [0], _neh, "流水车间 NEH 初始化", problems={"flowshop"}))
