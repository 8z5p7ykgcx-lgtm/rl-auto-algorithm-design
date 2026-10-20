"""Softmax 策略（numpy 实现，零重依赖）。

每个决策位置 i 维护独立的权重矩阵 W_i（shape: [state_dim, A_i]），
logits = x^T W_i，动作按 softmax 采样。REINFORCE 的梯度可直接计算。
"""

from __future__ import annotations

import numpy as np

from .env import DesignEnv


class SoftmaxPolicy:
    def __init__(self, env: DesignEnv, seed: int = 0, scale: float = 0.5):
        self.n_positions = len(env.positions)
        self.state_dim = env.state_dim()
        self.action_sizes = [env.action_space_size(i) for i in range(self.n_positions)]
        rng = np.random.default_rng(seed)
        self.W = [
            rng.normal(0.0, scale, size=(self.state_dim, a)) for a in self.action_sizes
        ]

    def logits(self, step: int, x: np.ndarray) -> np.ndarray:
        return x @ self.W[step]

    def action_probs(self, step: int, x: np.ndarray) -> np.ndarray:
        z = self.logits(step, x)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(self, step: int, x: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        """返回 (动作, 该动作的概率)。"""
        p = self.action_probs(step, x)
        a = rng.choice(len(p), p=p)
        return int(a), float(p[a])

    def grad_log_pi(self, step: int, x: np.ndarray, action: int) -> np.ndarray:
        """∇θ log π(a|x) 对第 step 位权重矩阵的梯度（shape: [state_dim, A_i]）。"""
        p = self.action_probs(step, x)
        g = np.zeros_like(self.W[step])
        g[:, action] = 1.0
        g -= np.outer(x, p)
        return g
