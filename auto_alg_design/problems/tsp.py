"""对称欧氏 TSP。解 = 城市访问顺序排列，目标 = 闭合回路总距离（最小化）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Problem, pairwise_euclidean


class TSP(Problem):
    name = "tsp"
    encoding = "permutation"

    def __init__(self, n: int = 30, seed: int | None = None):
        super().__init__(seed)
        self.n = n

    # ---- 实例 ----
    def random_instance(self, rng: np.random.Generator) -> dict[str, Any]:
        coords = rng.uniform(0, 1, size=(self.n, 2))
        return {"n": self.n, "coords": coords, "dist": pairwise_euclidean(coords)}

    # ---- 解与目标 ----
    def random_solution(self, instance: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        return rng.permutation(instance["n"])

    def cost(self, instance: dict[str, Any], solution: np.ndarray) -> float:
        dist = instance["dist"]
        idx = np.asarray(solution, dtype=int)
        return float(dist[idx, np.roll(idx, -1)].sum())

    # ---- 特征 ----
    def features(self, instance: dict[str, Any]) -> np.ndarray:
        dist = instance["dist"]
        n = instance["n"]
        upper = dist[np.triu_indices(n, k=1)]
        # 用近邻平均近似「最近邻边权」这一经典 TSP 地貌特征
        sorted_upper = np.sort(dist, axis=1)[:, 1:4]
        nn = sorted_upper.mean()
        return np.array(
            [n, upper.mean(), upper.std(), upper.min(), upper.max(), nn], dtype=float
        )

    # ---- 便捷方法（供可视化/贪心初始化使用）----
    @staticmethod
    def nearest_neighbor_tour(instance: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        """最近邻贪心构造初始回路。"""
        n = instance["n"]
        dist = instance["dist"]
        start = rng.integers(0, n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        cur = start
        while unvisited:
            nxt = min(unvisited, key=lambda c: dist[cur, c])
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return np.asarray(tour, dtype=int)
