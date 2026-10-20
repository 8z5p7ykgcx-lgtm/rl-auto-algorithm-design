"""REINFORCE（策略梯度 + 滑动平均基线）训练器。

流程：每 episode 采样一个新实例 → 策略逐位生成一个算法 → 运行求解 → 得到奖励
→ 用 (R - baseline) 作为优势更新各位置权重。训练若干轮后，策略学会为这类问题
「设计」出更高效的算法结构——即自动算法设计的核心。

采用带动量的 numpy SGD，避免引入 PyTorch 等重依赖，保证项目开箱即跑。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..problems.base import Problem
from .env import DesignEnv
from .features import estimate_feature_stats
from .policy import SoftmaxPolicy


@dataclass
class TrainConfig:
    episodes: int = 300
    lr: float = 0.1
    momentum: float = 0.9
    baseline_ema: float = 0.95      # 奖励基线指数滑动系数
    eval_every: int = 50            # 每 N episode 打印一次进度
    n_eval_instances: int = 5       # 进度评估用实例数
    n_eval_runs: int = 3            # 进度评估每个实例重复次数
    seed: int = 0
    log: bool = True


@dataclass
class TrainResult:
    policy: SoftmaxPolicy
    rewards: np.ndarray            # 每个 episode 的奖励
    eval_means: list[float]        # 每 eval_every 的评估平均奖励
    eval_steps: list[int]
    best_spec_text: str            # 最优算法描述（供展示）
    best_reward: float


def train(
    problem: Problem,
    cfg: TrainConfig | None = None,
    iterations: int = 60,
    n_search_steps: int = 4,
) -> TrainResult:
    cfg = cfg or TrainConfig()
    rng = np.random.default_rng(cfg.seed)

    # 特征标准化统计
    feats_mean, feats_std = estimate_feature_stats(problem, seed=cfg.seed)
    env = DesignEnv(
        problem=problem,
        instance=problem.random_instance(rng),
        feats_mean=feats_mean,
        feats_std=feats_std,
        n_search_steps=n_search_steps,
        iterations=iterations,
    )
    policy = SoftmaxPolicy(env, seed=cfg.seed)

    # 动量缓冲
    vel = [np.zeros_like(w) for w in policy.W]
    baseline = 0.0
    rewards = np.zeros(cfg.episodes)
    eval_means: list[float] = []
    eval_steps: list[int] = []
    best_reward = -1e9
    best_text = ""

    for ep in range(cfg.episodes):
        # 1) 采样实例
        inst = problem.random_instance(rng)
        env.instance = inst

        # 2) 策略逐位生成算法
        actions: list[int] = []
        probs: list[float] = []
        chosen: list[int] = []
        for step in range(len(env.positions)):
            x = env.state(step, chosen)
            a, p = policy.sample(step, x, rng)
            actions.append(a)
            probs.append(p)
            chosen.append(a)

        # 3) 运行评估
        spec = env.build_spec(actions, iterations=iterations)
        reward, _ = env.reward_for(spec, rng)
        rewards[ep] = reward
        if reward > best_reward:
            best_reward = reward
            best_text = spec.describe()

        # 4) 更新（REINFORCE with baseline）
        baseline = cfg.baseline_ema * baseline + (1 - cfg.baseline_ema) * reward
        advantage = reward - baseline
        chosen = []
        for step in range(len(env.positions)):
            x = env.state(step, chosen)
            g = policy.grad_log_pi(step, x, actions[step])
            vel[step] = cfg.momentum * vel[step] + cfg.lr * (advantage * g)
            policy.W[step] += vel[step]
            chosen.append(actions[step])

        # 5) 进度评估
        if cfg.log and (ep + 1) % cfg.eval_every == 0:
            m = _eval_mean_reward(policy, env, problem, cfg.n_eval_instances, cfg.n_eval_runs, rng)
            eval_means.append(m)
            eval_steps.append(ep + 1)
            print(f"[train] ep={ep + 1}/{cfg.episodes}  mean_reward={m:+.4f}  best={best_reward:+.4f}")

    return TrainResult(
        policy=policy,
        rewards=rewards,
        eval_means=eval_means,
        eval_steps=eval_steps,
        best_spec_text=best_text,
        best_reward=best_reward,
    )


def _eval_mean_reward(
    policy: SoftmaxPolicy,
    env: DesignEnv,
    problem: Problem,
    n_instances: int,
    n_runs: int,
    rng: np.random.Generator,
) -> float:
    """在若干未见实例上评估当前策略的平均奖励（用于训练进度展示）。"""
    total, cnt = 0.0, 0
    for _ in range(n_instances):
        inst = problem.random_instance(rng)
        env.instance = inst
        for _ in range(n_runs):
            actions, chosen = [], []
            for step in range(len(env.positions)):
                x = env.state(step, chosen)
                a, _ = policy.sample(step, x, rng)
                actions.append(a)
                chosen.append(a)
            spec = env.build_spec(actions)
            r, _ = env.reward_for(spec, rng)
            total += r
            cnt += 1
    return total / max(cnt, 1)


def generate_algorithm(policy: SoftmaxPolicy, env: DesignEnv, rng: np.random.Generator) -> AlgorithmSpec:
    """用训练好的策略为指定实例生成一个算法（评估阶段使用）。"""
    actions, chosen = [], []
    for step in range(len(env.positions)):
        x = env.state(step, chosen)
        a, _ = policy.sample(step, x, rng)
        actions.append(a)
        chosen.append(a)
    return env.build_spec(actions)
