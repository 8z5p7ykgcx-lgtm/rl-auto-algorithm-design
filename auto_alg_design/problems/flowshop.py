"""置换流水车间调度问题（PFSP）。解 = 工件排列，目标 = makespan（最小化）。

加工时间矩阵 P[j, m] = 工件 j 在机器 m 上的加工时间。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Problem


class FlowShop(Problem):
    name = "flowshop"
    encoding = "permutation"

    def __init__(self, n_jobs: int = 15, n_machines: int = 6, seed: int | None = None):
        super().__init__(seed)
        self.n_jobs = n_jobs
        self.n_machines = n_machines

    def random_instance(self, rng: np.random.Generator) -> dict[str, Any]:
        p = rng.integers(1, 100, size=(self.n_jobs, self.n_machines)).astype(float)
        return {"n_jobs": self.n_jobs, "n_machines": self.n_machines, "p": p}

    def random_solution(self, instance: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        return rng.permutation(instance["n_jobs"])

    @staticmethod
    def makespan_matrix(p: np.ndarray, order: np.ndarray) -> float:
        """按给定工件顺序计算 makespan。"""
        jobs = p[np.asarray(order, dtype=int)]  # (n, m)
        c = np.zeros(jobs.shape)
        c[0] = np.cumsum(jobs[0])
        for j in range(1, jobs.shape[0]):
            c[j, 0] = c[j - 1, 0] + jobs[j, 0]
            c[j, 1:] = np.maximum(c[j - 1, 1:], c[j, :-1]) + jobs[j, 1:]
        return float(c[-1, -1])

    def cost(self, instance: dict[str, Any], solution: np.ndarray) -> float:
        return self.makespan_matrix(instance["p"], solution)

    def features(self, instance: dict[str, Any]) -> np.ndarray:
        p = instance["p"]
        return np.array(
            [
                instance["n_jobs"], instance["n_machines"],
                p.mean(), p.std(), p.min(), p.max(),
                p.sum(axis=1).mean(), p.sum(axis=1).std(),
            ],
            dtype=float,
        )

    @staticmethod
    def neh_init(instance: dict[str, Any]) -> np.ndarray:
        """NEH 启发式构造初始排序（流水车间经典初始化器）。"""
        p = instance["p"]
        n = instance["n_jobs"]
        # 按总加工时间降序排列
        order = np.argsort(-p.sum(axis=1))
        seq: list[int] = []
        for job in order:
            best_pos, best_c = 0, float("inf")
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [int(job)] + seq[pos:]
                c = FlowShop.makespan_matrix(p, np.asarray(trial))
                if c < best_c:
                    best_c, best_pos = c, pos
            seq.insert(best_pos, int(job))
        return np.asarray(seq, dtype=int)
