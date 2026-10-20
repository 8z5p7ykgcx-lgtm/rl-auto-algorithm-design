"""算法设计环境：RL 智能体逐位选择构件与参数，组装出一个完整算法并评估。

决策模板（固定长度，便于策略梯度训练）：
    [init] -> [search×K] -> [accept]
每一步的动作 = (构件id, 参数档位)。生成完整算法后调用 runner 在实例上求解，
以「相对随机基线的改进率」作为奖励。这是方向1「一体化：构件组合 + 参数调优」的落点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..components.registry import AlgorithmSpec, Component, list_for
from ..components.runner import run_algorithm
from ..problems.base import Problem
from .features import estimate_feature_stats, normalize_features


@dataclass
class DesignEnv:
    """一次 episode 的环境状态与配置。"""

    problem: Problem
    instance: dict[str, Any]
    feats_mean: np.ndarray
    feats_std: np.ndarray
    n_search_steps: int = 4
    iterations: int = 60
    n_random_ref: int = 5                      # 随机基线解采样数

    positions: list[str] = field(default_factory=list)
    action_tables: list[list[tuple[str, int]]] = field(default_factory=list)  # 每位 [(cid, param)]
    total_actions: int = 0

    def __post_init__(self):
        enc = self.problem.encoding
        pname = self.problem.name
        inits = list_for("init", enc, pname)
        searches = list_for("search", enc, pname)
        accepts = list_for("accept", enc, pname)

        self.action_tables = []
        self.action_tables.append([(c.cid, p) for c in inits for p in c.param_values])
        for _ in range(self.n_search_steps):
            self.action_tables.append([(c.cid, p) for c in searches for p in c.param_values])
        self.action_tables.append([(c.cid, p) for c in accepts for p in c.param_values])

        self.positions = ["init"] + [f"search_{k}" for k in range(self.n_search_steps)] + ["accept"]
        self.total_actions = sum(len(t) for t in self.action_tables)

    # ---------- 状态 ----------
    def state(self, step: int, chosen: list[int]) -> np.ndarray:
        """step: 当前决策位索引(0..L-1)；chosen: 已选动作全局索引列表。"""
        raw = self.problem.features(self.instance)
        x = normalize_features(raw, self.feats_mean, self.feats_std)
        # 决策进度 one-hot
        prog = np.zeros(len(self.positions))
        prog[step] = 1.0
        # 已选动作累积向量（让策略"记得"自己拼到哪了）
        accum = np.zeros(self.total_actions)
        for a in chosen:
            accum[a] = 1.0
        return np.concatenate([x, prog, accum])

    def state_dim(self) -> int:
        return len(self.problem.features(self.instance)) + len(self.positions) + self.total_actions

    # ---------- 动作 ----------
    def action_space_size(self, step: int) -> int:
        return len(self.action_tables[step])

    def decode(self, step: int, action: int) -> tuple[str, int]:
        return self.action_tables[step][action]

    # ---------- 组装与评估 ----------
    def build_spec(self, actions: list[int], iterations: int | None = None) -> AlgorithmSpec:
        cid, p = self.decode(0, actions[0])
        spec = AlgorithmSpec(
            init_id=cid,
            init_param=p,
            search_steps=[self.decode(i + 1, actions[i + 1]) for i in range(self.n_search_steps)],
            accept_id=self.decode(len(actions) - 1, actions[-1])[0],
            accept_param=self.decode(len(actions) - 1, actions[-1])[1],
            iterations=iterations or self.iterations,
        )
        return spec

    def baseline_cost(self, rng: np.random.Generator) -> float:
        """随机解基线（评估「自动设计的算法比瞎搜好多少」的参照）。"""
        costs = [
            self.problem.cost(self.instance, self.problem.random_solution(self.instance, rng))
            for _ in range(self.n_random_ref)
        ]
        return float(np.mean(costs))

    def reward_for(self, spec: AlgorithmSpec, rng: np.random.Generator) -> tuple[float, float]:
        """运行生成的算法，返回 (奖励, 最终成本)。奖励 = 相对随机基线的改进率。

        分母取 ``abs(base)``：兼容背包这类负成本问题（base 为负时改进率仍为正值，
        避免接近 0 的负基数导致奖励尺度爆炸）。
        """
        base = self.baseline_cost(rng)
        cost, _, _ = run_algorithm(self.problem, self.instance, spec, rng)
        reward = (base - cost) / max(abs(base), 1e-9)
        return float(reward), float(cost)
