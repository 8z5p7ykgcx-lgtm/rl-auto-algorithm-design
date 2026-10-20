"""问题实例特征提取与标准化。

RL 的状态 = 实例轻量特征（对应综述中「基于问题特征做算法选择/设计」的思路）
+ 决策进度信息 + 已选构件轨迹。
"""

from __future__ import annotations

import numpy as np

from ..problems.base import Problem


def estimate_feature_stats(
    problem: Problem, n_samples: int = 40, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """采样若干实例估计特征均值与标准差（用于标准化）。"""
    rng = np.random.default_rng(seed)
    feats = []
    for _ in range(n_samples):
        inst = problem.random_instance(rng)
        feats.append(problem.features(inst))
    feats = np.asarray(feats)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def normalize_features(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(raw, dtype=float) - mean) / std
