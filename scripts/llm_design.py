#!/usr/bin/env python3
"""大模型驱动的自动算法设计：按模板给数据，LLM 自动选择算法去跑。

用法:
    # 先设置 API Key（DeepSeek 等 OpenAI 兼容服务）
    export LLM_API_KEY=sk-xxxx
    export LLM_BASE_URL=https://api.deepseek.com     # 可选，默认 DeepSeek
    export LLM_MODEL=deepseek-chat                   # 可选

    # 用模板跑（模板示例见 examples/）
    python scripts/llm_design.py --template examples/tsp.json

    # 没有 Key 时可用 --mock 本地演示完整流程
    python scripts/llm_design.py --template examples/tsp.json --mock

产出:
    outputs/llm_design_result.txt   算法设计 + 求解结果
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from auto_alg_design import FlowShop, Knapsack, TSP
from auto_alg_design.baselines.genetic_algorithm import GeneticAlgorithm
from auto_alg_design.baselines.simulated_annealing import SimulatedAnnealing
from auto_alg_design.components.runner import run_algorithm
from auto_alg_design.llm.client import LLMClient, load_config
from auto_alg_design.llm.designer import LLMDesignResult, design_with_llm
from auto_alg_design.problems.base import pairwise_euclidean

PROBLEM_FACTORIES = {
    "tsp": lambda seed: TSP(n=30, seed=seed),
    "knapsack": lambda seed: Knapsack(n=50, seed=seed),
    "flowshop": lambda seed: FlowShop(n_jobs=15, n_machines=6, seed=seed),
}


def load_problem_and_instance(problem_type: str, inst_data: dict, seed: int = 0):
    """按模板里的 instance 字段加载问题与实例。

    inst_data 支持三种形态：
    - 空/仅规模(n)：随机生成实例
    - tsp: {"coords": [[x,y],...]}
    - knapsack: {"values": [...], "weights": [...], "capacity": float}
    - flowshop: {"p": [[...], ...]}（n_jobs x n_machines 加工时间矩阵）
    """
    rng = np.random.default_rng(seed)
    if problem_type == "tsp":
        if "coords" in inst_data and inst_data["coords"]:
            coords = np.asarray(inst_data["coords"], dtype=float)
            problem = TSP(n=len(coords), seed=seed)
            instance = {"n": len(coords), "coords": coords, "dist": pairwise_euclidean(coords)}
        else:
            n = int(inst_data.get("n", 30))
            problem = TSP(n=n, seed=seed)
            instance = problem.random_instance(rng)
    elif problem_type == "knapsack":
        if "values" in inst_data and inst_data["values"]:
            values = np.asarray(inst_data["values"], dtype=float)
            weights = np.asarray(inst_data.get("weights", np.ones_like(values)), dtype=float)
            capacity = float(inst_data.get("capacity", 0.5 * weights.sum()))
            problem = Knapsack(n=len(values), seed=seed)
            instance = {"n": len(values), "values": values, "weights": weights, "capacity": capacity}
        else:
            n = int(inst_data.get("n", 50))
            problem = Knapsack(n=n, seed=seed)
            instance = problem.random_instance(rng)
    elif problem_type == "flowshop":
        if "p" in inst_data and inst_data["p"]:
            p = np.asarray(inst_data["p"], dtype=float)
            n_jobs, n_machines = p.shape
            problem = FlowShop(n_jobs=n_jobs, n_machines=n_machines, seed=seed)
            instance = {"n_jobs": n_jobs, "n_machines": n_machines, "p": p}
        else:
            n_jobs = int(inst_data.get("n_jobs", 15))
            n_machines = int(inst_data.get("n_machines", 6))
            problem = FlowShop(n_jobs=n_jobs, n_machines=n_machines, seed=seed)
            instance = problem.random_instance(rng)
    else:
        raise ValueError(f"不支持的 problem_type: {problem_type}（可选 tsp/knapsack/flowshop）")
    return problem, instance


class MockLLMClient:
    """本地 mock：按问题类型返回一个合理算法设计（无 API Key 时演示用）。"""

    def __init__(self, model="mock-designer"):
        self.model = model

    def is_configured(self) -> bool:
        return True

    def chat(self, messages, **kw):
        user = messages[-1]["content"]
        if "knapsack" in user:
            payload = {
                "action": "compose", "reasoning": "背包用价值密度贪心初始化+位翻转局部搜索+精英保留",
                "algorithm": {
                    "init": {"id": "greedy_kp", "param": 0},
                    "search_steps": [{"id": "bitflip_ls", "param": 1},
                                     {"id": "bitflip_mutation", "param": 1}],
                    "accept": {"id": "elite_accept", "param": 0},
                    "iterations": 60,
                },
            }
        elif "flowshop" in user:
            payload = {
                "action": "compose", "reasoning": "流水车间用 NEH 初始化+交换局部搜索+退火接受",
                "algorithm": {
                    "init": {"id": "neh_init", "param": 0},
                    "search_steps": [{"id": "first_improve_ls", "param": 1},
                                     {"id": "inversion_mutation", "param": 1}],
                    "accept": {"id": "sa_accept", "param": 1},
                    "iterations": 60,
                },
            }
        else:
            payload = {
                "action": "compose", "reasoning": "TSP 用最近邻初始化+2-opt局部搜索+精英保留",
                "algorithm": {
                    "init": {"id": "greedy_tsp", "param": 0},
                    "search_steps": [{"id": "two_opt_ls", "param": 1},
                                     {"id": "swap_mutation", "param": 0}],
                    "accept": {"id": "elite_accept", "param": 0},
                    "iterations": 60,
                },
            }
        return json.dumps(payload, ensure_ascii=False)


def run_with_spec(problem, instance, spec, runs: int, seed: int = 0):
    """把 LLM 设计的算法跑 runs 次，返回 (最好成本, 平均成本, 轨迹最好一次)。"""
    best_cost, costs, best_traj = float("inf"), [], None
    for r in range(runs):
        rng = np.random.default_rng(seed + r)
        cost, traj, sol = run_algorithm(problem, instance, spec, rng, record_trajectory=True)
        costs.append(cost)
        if cost < best_cost:
            best_cost, best_traj = cost, traj
    return best_cost, float(np.mean(costs)), best_traj


def main():
    ap = argparse.ArgumentParser(description="LLM 驱动的自动算法设计")
    ap.add_argument("--template", required=True, help="模板 JSON 路径（见 examples/）")
    ap.add_argument("--key", default=None, help="API Key（优先用环境变量 LLM_API_KEY）")
    ap.add_argument("--base-url", default=None, help="OpenAI 兼容 base_url（默认 DeepSeek）")
    ap.add_argument("--model", default=None, help="模型名（默认 deepseek-chat）")
    ap.add_argument("--runs", type=int, default=3, help="对设计出的算法重复运行次数")
    ap.add_argument("--baselines", action="store_true", help="同时对比 GA / SA / Random")
    ap.add_argument("--mock", action="store_true", help="使用本地 mock（无需 API Key，演示流程）")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    # 读取模板
    with open(args.template, "r", encoding="utf-8") as f:
        tmpl = json.load(f)
    problem_type = tmpl.get("problem_type", "tsp")
    inst_data = tmpl.get("instance", {})
    iterations = int(tmpl.get("iterations", 60))
    n_search_steps = int(tmpl.get("n_search_steps", 4))
    runs = int(tmpl.get("runs", args.runs))
    llm_cfg = tmpl.get("llm", {})
    model = args.model or llm_cfg.get("model")
    temperature = llm_cfg.get("temperature")

    problem, instance = load_problem_and_instance(problem_type, inst_data, seed=0)
    os.makedirs(args.out, exist_ok=True)

    # 调用大模型设计算法
    if args.mock:
        client = MockLLMClient(model=model or "mock-designer")
        print(f"[llm] mock 模式：本地规则生成算法设计（不调用真实 API）")
    else:
        config = load_config(api_key=args.key, base_url=args.base_url, model=model)
        client = LLMClient(config)
        if not client.is_configured():
            print("[llm] 未检测到 LLM_API_KEY。")
            print("      设置方式: export LLM_API_KEY=sk-xxxx  （可选 LLM_BASE_URL / LLM_MODEL）")
            print("      或加 --mock 参数本地演示。")
            sys.exit(1)

    t0 = time.time()
    result: LLMDesignResult = design_with_llm(
        client, problem, instance, n_search_steps=n_search_steps,
        default_iterations=iterations, temperature=temperature,
    )
    dt = time.time() - t0

    # 执行设计出的算法
    best_cost, mean_cost, _ = run_with_spec(problem, instance, result.spec, runs, seed=1)

    lines = [
        "=" * 70,
        f"LLM 驱动的自动算法设计  问题={problem.name}  模型={getattr(client, 'model', '?')}",
        "=" * 70,
        f"实例特征: {problem.features(instance)}",
        f"LLM 设计的算法: {result.spec.describe()}",
        f"决策依据(reasoning/原始回复): {result.raw_response.strip()[:400]}",
    ]
    if result.error:
        lines.append(f"[警告] LLM 输出校验失败，已回退默认算法: {result.error}")
    if result.used_fallback:
        lines.append("[警告] 使用了回退算法（LLM 输出不可用）")
    lines += [
        f"求解结果:  {runs} 次运行  最优成本={best_cost:.4f}  平均成本={mean_cost:.4f}",
        f"调用耗时:  {dt:.2f}s",
    ]

    # 可选的基线对比
    if args.baselines:
        print(f"\n[bench] 对比 GA / SA / Random ...")
        rng = np.random.default_rng(2)
        random_cost = float(np.mean([
            problem.cost(instance, problem.random_solution(instance, rng)) for _ in range(20)
        ]))
        ga = GeneticAlgorithm(population=24, generations=40, seed=3)
        ga_sol, ga_cost = ga.solve(problem, instance)
        sa = SimulatedAnnealing(iterations=600, seed=5)
        sa_sol, sa_cost = sa.solve(problem, instance)
        lines += [
            "",
            "----- 对比基线 -----",
            f"{'方法':<16}{'成本':>12}   {'相对随机改进':>12}",
            f"{'LLM-designed':<16}{best_cost:>12.4f}   {100 * (random_cost - best_cost) / max(abs(random_cost), 1e-9):>+10.2f}%",
            f"{'GA':<16}{ga_cost:>12.4f}   {100 * (random_cost - ga_cost) / max(abs(random_cost), 1e-9):>+10.2f}%",
            f"{'SA':<16}{sa_cost:>12.4f}   {100 * (random_cost - sa_cost) / max(abs(random_cost), 1e-9):>+10.2f}%",
            f"{'Random':<16}{random_cost:>12.4f}   {0.0:>+10.2f}%",
        ]

    out_path = os.path.join(args.out, "llm_design_result.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[llm] 结果已保存 -> {out_path}")


if __name__ == "__main__":
    main()
