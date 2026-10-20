"""接受/更新构件：决定候选解是否被接受。

- greedy_accept    只接受严格改进的解（登山式）
- elite_accept     维护全局最优，候选若优于历史最优才替换（精英保留）
- sa_accept        模拟退火式接受（带温度档位，允许一定概率接受劣解以逃逸局部最优）

接受构件返回 (接受后的当前解, 更新后的记忆状态)；记忆状态用于 elite/sa 的跨步信息。
"""

from __future__ import annotations

import numpy as np

from ..problems.base import Problem
from .registry import Component, register


def _greedy_accept(problem, instance, current, candidate, rng, param, memory=None):
    if problem.cost(instance, candidate) < problem.cost(instance, current) - 1e-9:
        return candidate, None
    return current, None


def _elite_accept(problem, instance, current, candidate, rng, param, memory=None):
    memory = memory if memory is not None else {"best": current.copy()}
    if problem.cost(instance, candidate) < problem.cost(instance, memory["best"]) - 1e-9:
        memory["best"] = candidate.copy()
        return candidate, memory
    return current, memory


def _sa_accept(problem, instance, current, candidate, rng, param, memory=None):
    memory = memory if memory is not None else {"best": current.copy(), "t0": None}
    c_cur = problem.cost(instance, current)
    c_cand = problem.cost(instance, candidate)
    if c_cand < c_cur - 1e-9:
        if c_cand < problem.cost(instance, memory["best"]) - 1e-9:
            memory["best"] = candidate.copy()
        return candidate, memory
    # 温度随迭代衰减：由 runner 把迭代进度注入 memory["progress"]∈[0,1]
    if memory.get("t0") is None:
        memory["t0"] = max(c_cur, 1e-9)
    T = memory["t0"] * [0.1, 0.3, 0.8][param] * (1.0 - memory.get("progress", 0.0))
    if rng.random() < np.exp(-(c_cand - c_cur) / max(T, 1e-9)):
        return candidate, memory
    return current, memory


register(Component("greedy_accept", "accept", "any", [0], _greedy_accept, "贪心接受"))
register(Component("elite_accept", "accept", "any", [0], _elite_accept, "精英保留"))
register(Component("sa_accept", "accept", "any", [0, 1, 2], _sa_accept, "退火式接受"))
