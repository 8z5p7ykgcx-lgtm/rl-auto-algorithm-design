"""可视化：训练曲线、评估对比、TSP 路径图。输出到 outputs/ 目录。"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无界面后端，适合脚本/服务器
import matplotlib.pyplot as plt

from .evaluate import EvalResult

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "WenQuanYi Micro Hei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved -> {path}")


def plot_training_curve(rewards: np.ndarray, eval_steps: list[int], eval_means: list[float], path: str):
    """训练奖励曲线（含评估点）。"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rewards, alpha=0.35, lw=0.8, label="episode reward")
    # 滑动平均
    win = max(1, len(rewards) // 30)
    ma = np.convolve(rewards, np.ones(win) / win, mode="valid")
    ax.plot(np.arange(win - 1, len(rewards)), ma, lw=2, color="C1", label="moving average")
    if eval_steps:
        ax.plot(eval_steps, eval_means, "o--", color="C3", label="hold-out eval")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward (improvement over random)")
    ax.set_title("RL training curve: auto algorithm design")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, path)


def plot_comparison(res: EvalResult, path: str):
    """各方法平均成本对比条形图。"""
    names = [m.name for m in res.methods]
    means = [m.mean_cost for m in res.methods]
    stds = [m.std_cost for m in res.methods]
    colors = ["#2e7d32", "#1565c0", "#ef6c00", "#757575"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean cost (lower is better)")
    ax.set_title(f"Algorithm comparison on {res.problem_name} (unseen instances)")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, path)


def plot_tsp_tour(instance: dict, solutions: dict[str, np.ndarray], path: str):
    """TSP 路径可视化：各算法求得的路径对比。"""
    coords = instance["coords"]
    fig, axes = plt.subplots(1, len(solutions), figsize=(4.6 * len(solutions), 4.2), squeeze=False)
    for ax, (name, sol) in zip(axes[0], solutions.items()):
        pts = coords[np.asarray(sol)]
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], "o-", ms=4, lw=1.2)
        ax.set_title(name)
        ax.set_aspect("equal")
    fig.suptitle("TSP routes found by different algorithms")
    save_fig(fig, path)


def plot_spec_usage(specs: list[str], path: str):
    """统计 RL 自动设计出的算法中构件使用频率（去重算法）。"""
    from collections import Counter

    counter: Counter = Counter()
    for s in specs:
        for token in s.split(" -> "):
            cid = token.split("#")[0].replace("init=", "").replace("accept=", "")
            counter[cid] += 1
    items = counter.most_common()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#3949ab")
    ax.set_xlabel("usage count in designed algorithms")
    ax.set_title("Component usage in RL-designed algorithms")
    save_fig(fig, path)
