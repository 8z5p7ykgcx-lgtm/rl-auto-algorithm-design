"""问题基类与统一接口。

所有组合优化问题都实现同一个最小接口，使得「算法构件库 / RL环境 / 评估器」
可以完全与具体问题解耦。本设计对应综述中「问题实例 → 算法自动设计」的解耦思想。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Problem(ABC):
    """组合优化问题基类。

    统一约定（重要）：
    - ``cost`` 一律为**最小化**目标（TSP=总距离, 背包=负总价值, 流水车间=makespan）。
    - 解 ``solution`` 是 numpy 数组：排列编码（TSP/流水车间）或 0/1 编码（背包）。
    """

    name: str = "base"
    encoding: str = "permutation"  # "permutation" | "binary"

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ---- 实例生成与信息 ----
    @abstractmethod
    def random_instance(self, rng: np.random.Generator) -> dict[str, Any]:
        """生成一个随机问题实例（特征矩阵、约束等）。"""

    # ---- 解与目标 ----
    @abstractmethod
    def random_solution(self, instance: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        """随机生成一个可行解。"""

    @abstractmethod
    def cost(self, instance: dict[str, Any], solution: np.ndarray) -> float:
        """计算解的成本（最小化目标）。"""

    @abstractmethod
    def features(self, instance: dict[str, Any]) -> np.ndarray:
        """提取实例的轻量数值特征（供 RL 状态使用）。"""

    def repair(self, instance: dict[str, Any], solution: np.ndarray) -> np.ndarray:
        """修复非法解；默认不改动（排列编码恒可行，背包需覆盖实现）。"""
        return solution


def pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    """由二维坐标计算欧氏距离矩阵。"""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
