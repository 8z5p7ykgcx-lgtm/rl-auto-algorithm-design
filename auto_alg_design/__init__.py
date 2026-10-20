"""智能优化算法自动设计（方向1验证原型）。

用强化学习从构件库中自动组合算法并调参，在 TSP / 背包 / 流水车间
三个经典组合优化问题上，对比人工设计的 GA、SA。
"""

__version__ = "0.1.0"

from .problems.base import Problem
from .problems.tsp import TSP
from .problems.knapsack import Knapsack
from .problems.flowshop import FlowShop
from .components.registry import AlgorithmSpec
from .rl.env import DesignEnv
from .rl.policy import SoftmaxPolicy
from .rl.trainer import train, generate_algorithm, TrainConfig, TrainResult

__all__ = [
    "Problem", "TSP", "Knapsack", "FlowShop",
    "AlgorithmSpec", "DesignEnv", "SoftmaxPolicy",
    "train", "generate_algorithm", "TrainConfig", "TrainResult",
    "__version__",
]
