"""模拟退火（SA）基线：经典单解局部搜索框架。"""

from __future__ import annotations

import numpy as np

from ..problems.base import Problem


class SimulatedAnnealing:
    def __init__(
        self,
        iterations: int = 60,
        t0: float = 0.5,
        tf: float = 0.001,
        seed: int = 0,
    ):
        self.iterations = iterations
        self.t0 = t0
        self.tf = tf
        self.rng = np.random.default_rng(seed)

    def _neighbor(self, problem: Problem, instance: dict, sol: np.ndarray) -> np.ndarray:
        if problem.encoding == "permutation":
            n = len(sol)
            i, j = self.rng.choice(n, size=2, replace=False)
            s = sol.copy()
            s[i], s[j] = s[j], s[i]
            return s
        idx = int(self.rng.integers(0, len(sol)))
        s = sol.copy()
        s[idx] = 1 - s[idx]
        return problem.repair(instance, s)

    def solve(self, problem: Problem, instance: dict) -> tuple[np.ndarray, float]:
        cur = problem.random_solution(instance, self.rng)
        cur_c = problem.cost(instance, cur)
        best, best_c = cur.copy(), cur_c

        for t in range(self.iterations):
            temp = self.t0 + (self.tf - self.t0) * (t / max(self.iterations - 1, 1))
            nbr = self._neighbor(problem, instance, cur)
            nbr_c = problem.cost(instance, nbr)
            if nbr_c < cur_c or self.rng.random() < np.exp(-(nbr_c - cur_c) / max(temp, 1e-9)):
                cur, cur_c = nbr, nbr_c
                if cur_c < best_c:
                    best, best_c = cur.copy(), cur_c
        return best, best_c
