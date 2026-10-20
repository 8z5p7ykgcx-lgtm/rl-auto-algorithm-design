"""搜索构件：扰动算子与局部搜索算子。

- swap_mutation      随机交换（排列编码）
- inversion_mutation 区间逆序（排列编码）
- insert_mutation    随机插入（排列编码）
- bitflip_mutation   随机位翻转 + 修复（二进制编码）
- two_opt_ls         TSP 2-opt 局部搜索
- first_improve_ls   排列 first-improvement 交换局部搜索
- bitflip_ls         背包位翻转 first-improvement 局部搜索

每个构件带离散参数档位（param 索引），由 RL 一并选择——这即「构件+参数」联合决策。
"""

from __future__ import annotations

import numpy as np

from ..problems.base import Problem
from .registry import Component, register


# ---------- 扰动算子 ----------

def _swap(problem, instance, sol, rng, param) -> np.ndarray:
    n = len(sol)
    s = sol.copy()
    k = [1, 2, 4][param]
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        s[i], s[j] = s[j], s[i]
    return s


def _inversion(problem, instance, sol, rng, param) -> np.ndarray:
    n = len(sol)
    s = sol.copy()
    seg = max(2, int(n * [0.2, 0.4, 0.6][param]))
    start = rng.integers(0, n - seg + 1)
    s[start:start + seg] = s[start:start + seg][::-1]
    return s


def _insert(problem, instance, sol, rng, param) -> np.ndarray:
    n = len(sol)
    s = sol.copy()
    k = [1, 2][param]
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        val = s[i]
        s = np.delete(s, i)
        s = np.insert(s, j, val)
    return s


def _bitflip(problem, instance, sol, rng, param) -> np.ndarray:
    n = len(sol)
    s = sol.copy()
    k = [1, 3, 6][param]
    idx = rng.choice(n, size=k, replace=False)
    s[idx] = 1 - s[idx]
    return problem.repair(instance, s)


# ---------- 局部搜索算子 ----------

def _two_opt(problem, instance, sol, rng, param) -> np.ndarray:
    """2-opt：随机起点、first-improvement，扫描若干轮。"""
    dist = instance["dist"]
    s = np.asarray(sol, dtype=int).copy()
    n = len(s)
    rounds = [2, 5, 10][param]
    improved = True
    while improved:
        improved = False
        for _ in range(rounds):
            i, j = sorted(rng.choice(n, size=2, replace=False))
            if i == 0:
                continue
            a, b = s[i - 1], s[i]
            c, d = s[j], s[(j + 1) % n]
            delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
            if delta < -1e-9:
                s[i:j + 1] = s[i:j + 1][::-1]
                improved = True
    return s


def _first_improve_swap(problem, instance, sol, rng, param) -> np.ndarray:
    """排列编码：first-improvement 交换局部搜索（扫描 rounds 轮）。"""
    s = np.asarray(sol, dtype=int).copy()
    n = len(s)
    rounds = [2, 5, 10][param]
    improved = True
    while improved:
        improved = False
        for _ in range(rounds):
            i, j = sorted(rng.choice(n, size=2, replace=False))
            cand = s.copy()
            cand[i], cand[j] = cand[j], cand[i]
            if problem.cost(instance, cand) < problem.cost(instance, s) - 1e-9:
                s = cand
                improved = True
    return s


def _bitflip_ls(problem, instance, sol, rng, param) -> np.ndarray:
    """背包：单点翻转 first-improvement 局部搜索。"""
    s = np.asarray(sol, dtype=int).copy()
    n = len(s)
    rounds = [3, 8, 15][param]
    improved = True
    while improved:
        improved = False
        for _ in range(rounds):
            i = int(rng.integers(0, n))
            cand = s.copy()
            cand[i] = 1 - cand[i]
            cand = problem.repair(instance, cand)
            if problem.cost(instance, cand) < problem.cost(instance, s) - 1e-9:
                s = cand
                improved = True
    return s


register(Component("swap_mutation", "search", "permutation", [0, 1, 2], _swap, "随机交换扰动"))
register(Component("inversion_mutation", "search", "permutation", [0, 1, 2], _inversion, "区间逆序扰动"))
register(Component("insert_mutation", "search", "permutation", [0, 1], _insert, "随机插入扰动"))
register(Component("bitflip_mutation", "search", "binary", [0, 1, 2], _bitflip, "位翻转扰动"))
register(Component("two_opt_ls", "search", "permutation", [0, 1, 2], _two_opt, "2-opt 局部搜索", problems={"tsp"}))
register(Component("first_improve_ls", "search", "permutation", [0, 1, 2], _first_improve_swap, "交换局部搜索"))
register(Component("bitflip_ls", "search", "binary", [0, 1, 2], _bitflip_ls, "位翻转局部搜索"))
