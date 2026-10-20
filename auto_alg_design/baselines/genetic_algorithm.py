"""手工设计的传统元启发式基线：遗传算法（GA）。

作用：作为「人工设计算法」的对照，验证 RL 自动设计出的算法能否达到/超越
人工设计的经典算法。GA 按问题编码分排列/二进制两套算子。
"""

from __future__ import annotations

import numpy as np

from ..problems.base import Problem


class GeneticAlgorithm:
    def __init__(
        self,
        population: int = 24,
        generations: int = 40,
        mutation_rate: float = 0.3,
        elite_size: int = 2,
        tournament: int = 3,
        seed: int = 0,
    ):
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament = tournament
        self.rng = np.random.default_rng(seed)

    def _random_ind(self, problem: Problem, instance: dict) -> np.ndarray:
        return problem.random_solution(instance, self.rng)

    def _crossover(self, problem: Problem, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if problem.encoding == "permutation":
            return self._ox_crossover(a, b)
        return self._binary_crossover(a, b)

    def _ox_crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """顺序交叉（OX）：保留 a 的一段，其余按 b 的顺序补齐。"""
        n = len(a)
        i, j = sorted(self.rng.choice(n, size=2, replace=False))
        child = np.full(n, -1, dtype=int)
        child[i:j + 1] = a[i:j + 1]
        rest = [x for x in b if x not in child]
        pos = 0
        for k in range(n):
            if child[k] == -1:
                child[k] = rest[pos]
                pos += 1
        return child

    @staticmethod
    def _binary_crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = np.random.default_rng().integers(0, 2, size=len(a)).astype(bool)
        return np.where(mask, a, b)

    def _mutate(self, problem: Problem, instance: dict, ind: np.ndarray) -> np.ndarray:
        if problem.encoding == "permutation":
            if self.rng.random() < self.mutation_rate:
                i, j = self.rng.choice(len(ind), size=2, replace=False)
                ind = ind.copy()
                ind[i], ind[j] = ind[j], ind[i]
        else:
            n = len(ind)
            k = max(1, int(self.mutation_rate * n))
            idx = self.rng.choice(n, size=k, replace=False)
            ind = ind.copy()
            ind[idx] = 1 - ind[idx]
        return problem.repair(instance, ind)

    def _select(self, problem: Problem, instance: dict, pop: list[np.ndarray]) -> np.ndarray:
        best = None
        best_c = float("inf")
        for _ in range(self.tournament):
            cand = pop[int(self.rng.integers(len(pop)))]
            c = problem.cost(instance, cand)
            if c < best_c:
                best_c, best = c, cand
        return best

    def solve(self, problem: Problem, instance: dict) -> tuple[np.ndarray, float]:
        """返回 (最优解, 最优成本)。"""
        pop = [self._random_ind(problem, instance) for _ in range(self.population)]
        costs = np.array([problem.cost(instance, p) for p in pop])
        best = pop[int(np.argmin(costs))].copy()
        best_c = float(costs.min())

        for _ in range(self.generations):
            elite = sorted(zip(costs, pop), key=lambda t: t[0])[: self.elite_size]
            new_pop = [p.copy() for _, p in elite]
            while len(new_pop) < self.population:
                p1 = self._select(problem, instance, pop)
                p2 = self._select(problem, instance, pop)
                child = self._crossover(problem, p1, p2)
                child = self._mutate(problem, instance, child)
                child = problem.repair(instance, child)
                new_pop.append(child)
            pop = new_pop
            costs = np.array([problem.cost(instance, p) for p in pop])
            if costs.min() < best_c:
                best_c = float(costs.min())
                best = pop[int(np.argmin(costs))].copy()
        return best, best_c
