"""0-1 背包问题。解 = 0/1 数组，目标 = 负总价值（最小化形式），容量约束需修复。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Problem


class Knapsack(Problem):
    name = "knapsack"
    encoding = "binary"

    def __init__(self, n: int = 50, capacity_ratio: float = 0.5, seed: int | None = None):
        super().__init__(seed)
        self.n = n
        self.capacity_ratio = capacity_ratio

    def random_instance(self, rng: np.random.Generator) -> dict[str, Any]:
        values = rng.uniform(1, 100, size=self.n)
        weights = rng.uniform(1, 50, size=self.n)
        capacity = self.capacity_ratio * weights.sum()
        return {
            "n": self.n,
            "values": values,
            "weights": weights,
            "capacity": capacity,
        }

    def random_solution(self, instance: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        sol = rng.integers(0, 2, size=self.n).astype(int)
        return self.repair(instance, sol)

    def repair(self, instance: dict[str, Any], solution: np.ndarray) -> np.ndarray:
        """按价值密度从低到高剔除物品，直到满足容量约束。"""
        sol = np.asarray(solution, dtype=int).copy()
        weights = instance["weights"]
        values = instance["values"]
        capacity = instance["capacity"]
        total_w = sol @ weights
        if total_w <= capacity:
            return sol
        # 按价值密度升序剔除已选物品
        density = values / weights
        selected = np.where(sol == 1)[0]
        order = selected[np.argsort(density[selected])]
        for i in order:
            if total_w <= capacity:
                break
            sol[i] = 0
            total_w -= weights[i]
        return sol

    def cost(self, instance: dict[str, Any], solution: np.ndarray) -> float:
        sol = self.repair(instance, solution)
        return float(-(sol @ instance["values"]))

    def features(self, instance: dict[str, Any]) -> np.ndarray:
        v = instance["values"]
        w = instance["weights"]
        density = v / w
        return np.array(
            [
                instance["n"],
                v.mean(), v.std(), v.min(), v.max(),
                w.mean(), w.std(),
                density.mean(), density.std(),
                instance["capacity"] / w.sum(),
            ],
            dtype=float,
        )
