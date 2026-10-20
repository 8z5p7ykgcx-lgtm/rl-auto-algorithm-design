"""LLM 算法设计器：构建 prompt、解析并校验大模型的算法设计决策。

大模型不直接求解数值问题（那是 runner + 构件库的职责），它只做「决策」：
根据问题类型、实例规模与统计特征，从构件库中选出一套算法组合（或直接选择
现成算法）。本模块负责：

1. 把实例特征 + 构件库清单组织成结构化 prompt
2. 解析大模型输出的 JSON（容忍 ```json 代码块包裹等常见杂讯）
3. 逐项校验构件 id / 参数档位 / 结构合法性，非法时回退默认算法
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from ..components.registry import AlgorithmSpec, get, list_for
from ..problems.base import Problem
from .client import LLMClient

# ---------- 输出约束 ----------

SYSTEM_PROMPT = """你是智能优化算法设计专家。系统提供一个算法构件库，包含三类构件：
- init：初始化解生成方式
- search：搜索/扰动算子（会循环使用，param 为档位索引，值越大算子强度越高）
- accept：候选解接受规则

你的任务：根据给定的问题类型、实例规模与统计特征，从构件库中选择并组合出一套
最合适的算法。要求：
1. init 构件必须适用于该问题的编码方式；
2. search_steps 给出 1~4 个搜索构件（可重复），它们会按顺序循环执行；
3. accept 构件控制接受策略；
4. 若实例规模大/特征复杂，可适当提高算子强度档位或迭代次数。

只输出一个 JSON 对象，不要输出任何其他文字。格式严格如下：
{
  "action": "compose",
  "reasoning": "一句话说明选型理由",
  "algorithm": {
    "init": {"id": "构件id", "param": 整数档位},
    "search_steps": [{"id": "构件id", "param": 整数档位}, ...最多4个],
    "accept": {"id": "构件id", "param": 整数档位},
    "iterations": 整数迭代次数(10~300)
  }
}
其中构件 id 必须从给定的构件清单中选择，param 必须在清单给出的档位范围内。"""


# ---------- prompt 构建 ----------

def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def _component_catalog(problem_name: str, encoding: str) -> str:
    lines = ["【可用构件清单】"]
    for kind, title in [("init", "init 初始化"), ("search", "search 搜索/扰动"), ("accept", "accept 接受规则")]:
        comps = list_for(kind, encoding, problem_name)
        for c in comps:
            params = ", ".join(str(i) for i in range(len(c.param_values)))
            scope = f"仅适用于: {sorted(c.problems)}" if c.problems else "通用"
            lines.append(f"- {c.cid} ({title}, 档位范围 [{params}], {scope}): {c.desc}")
    return "\n".join(lines)


# 各问题的实例特征名（与 problems 模块 features() 的维度一一对应）
FEATURE_NAMES: dict[str, list[str]] = {
    "tsp": ["n_cities", "dist_mean", "dist_std", "dist_min", "dist_max", "nearest_neighbor_mean"],
    "knapsack": ["n_items", "value_mean", "value_std", "value_min", "value_max",
                 "weight_mean", "weight_std", "density_mean", "density_std", "capacity_ratio"],
    "flowshop": ["n_jobs", "n_machines", "p_mean", "p_std", "p_min", "p_max",
                 "row_sum_mean", "row_sum_std"],
}


def _feature_names(problem: Problem) -> list[str]:
    names = FEATURE_NAMES.get(problem.name)
    if names is None:
        names = [f"feat{i}" for i in range(len(problem.features(problem.random_instance(np.random.default_rng(0)))))]
    return names


def build_user_prompt(
    problem: Problem,
    instance: dict,
    n_search_steps: int = 4,
    default_iterations: int = 60,
) -> str:
    """把实例统计特征（而非原始数据）发送给大模型做决策。"""
    feats = np.asarray(problem.features(instance), dtype=float)
    feat_names = _feature_names(problem)
    feat_str = "; ".join(f"{n}={v:.4f}" for n, v in zip(feat_names, feats))
    encoding = "permutation" if problem.name != "knapsack" else "binary"
    n_dim = instance.get("n", "?")
    return (
        f"【问题类型】{problem.name}\n"
        f"【实例规模】决策变量维度 = {n_dim}\n"
        f"【实例统计特征】{feat_str}\n"
        f"【搜索步预算】search_steps 最多 {n_search_steps} 步；主循环迭代默认 {default_iterations}（可在 algorithm.iterations 中调整）\n"
        f"{_component_catalog(problem.name, encoding)}\n"
        f"\n请按系统提示的 JSON 格式输出你的算法设计。"
    )


# ---------- 响应解析 ----------

def _extract_json(text: str) -> dict:
    """从模型回复中提取 JSON 对象（容忍代码块、前后缀杂讯）。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("回复中未找到 JSON 对象")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败：{e}") from e


def _normalize_algo_json(data: dict) -> tuple[str, dict | str]:
    """校验顶层结构，返回 (action, algorithm)。"""
    action = data.get("action", "compose")
    algo = data.get("algorithm")
    if action not in {"compose", "select"}:
        raise ValueError(f"action 只能是 compose 或 select，得到: {action!r}")
    if algo is None:
        raise ValueError("缺少 algorithm 字段")
    return action, algo


def validate_compose(
    algo: dict,
    problem_name: str,
    encoding: str,
    max_search_steps: int,
) -> tuple[AlgorithmSpec | None, list[str]]:
    """校验 compose 结构，返回 (AlgorithmSpec, 错误列表)。错误非空则 spec 为 None。"""
    errors: list[str] = []

    def pick_comp(kind: str, entry, allow_missing: bool = False):
        if not isinstance(entry, dict):
            if allow_missing:
                return None
            errors.append(f"{kind} 必须是对象 {{id, param}}")
            return None
        cid = entry.get("id")
        param = entry.get("param", 0)
        if not isinstance(cid, str) or cid not in {c.cid for c in list_for(kind, encoding, problem_name)}:
            errors.append(f"{kind} 构件 {cid!r} 不在当前问题可用清单中")
            return None
        comp = get(cid)
        if not isinstance(param, int) or not (0 <= param < len(comp.param_values)):
            errors.append(f"{kind} 构件 {cid} 的 param={param!r} 超出档位范围 [0, {len(comp.param_values) - 1}]")
            return None
        return comp, param

    init = pick_comp("init", algo.get("init"))
    accept = pick_comp("accept", algo.get("accept"))

    steps_raw = algo.get("search_steps", [])
    steps: list[tuple[str, int]] = []
    if not isinstance(steps_raw, list) or len(steps_raw) < 1:
        errors.append("search_steps 必须是非空列表（1~4 个搜索构件）")
    else:
        if len(steps_raw) > max_search_steps:
            steps_raw = steps_raw[:max_search_steps]
        for s in steps_raw:
            picked = pick_comp("search", s)
            if picked is not None:
                steps.append((picked[0].cid, picked[1]))

    iters = algo.get("iterations", 60)
    if not isinstance(iters, int) or not (10 <= iters <= 300):
        errors.append(f"iterations={iters!r} 必须在 10~300 之间")

    if errors:
        return None, errors
    return AlgorithmSpec(
        init_id=init[0].cid, init_param=init[1],
        search_steps=steps, accept_id=accept[0].cid, accept_param=accept[1],
        iterations=iters,
        meta={"source": "llm"},
    ), []


def default_spec(problem_name: str, encoding: str, iterations: int = 60) -> AlgorithmSpec:
    """非法响应时的回退算法：贪心/随机初始化 + 局部搜索 + 精英保留。"""
    if encoding == "binary":
        return AlgorithmSpec(
            init_id="greedy_kp", init_param=0,
            search_steps=[("bitflip_ls", 1), ("bitflip_mutation", 1)],
            accept_id="elite_accept", accept_param=0, iterations=iterations,
            meta={"source": "fallback"},
        )
    init = "greedy_tsp" if problem_name == "tsp" else "neh_init" if problem_name == "flowshop" else "random_init"
    return AlgorithmSpec(
        init_id=init, init_param=0,
        search_steps=[("first_improve_ls", 1), ("insert_mutation", 0)],
        accept_id="elite_accept", accept_param=0, iterations=iterations,
        meta={"source": "fallback"},
    )


def spec_from_llm_json(
    data: dict,
    problem_name: str,
    encoding: str,
    max_search_steps: int = 4,
    default_iterations: int = 60,
) -> tuple[AlgorithmSpec | None, str]:
    """把大模型 JSON 决策转成 AlgorithmSpec。

    返回 (spec, error)；spec 为 None 且 error 非空表示解析/校验失败。
    """
    try:
        action, algo = _normalize_algo_json(data)
    except ValueError as e:
        return None, str(e)

    if action == "select":
        name = algo if isinstance(algo, str) else algo.get("name", "")
        if name not in {"ga", "sa", "random"}:
            return None, f"select 只能选 ga/sa/random，得到 {name!r}"
        return AlgorithmSpec(
            init_id="random_init", init_param=0,
            search_steps=[("first_improve_ls", 0)],
            accept_id="elite_accept", accept_param=0,
            iterations=default_iterations,
            meta={"source": "llm", "select": name},
        ), ""

    spec, errors = validate_compose(algo, problem_name, encoding, max_search_steps)
    if errors:
        return None, "; ".join(errors)
    return spec, ""


@dataclass
class LLMDesignResult:
    spec: AlgorithmSpec
    raw_response: str
    error: str = ""
    used_fallback: bool = False
    parsed: dict = field(default_factory=dict)


def design_with_llm(
    client: LLMClient,
    problem: Problem,
    instance: dict,
    n_search_steps: int = 4,
    default_iterations: int = 60,
    temperature: float | None = None,
) -> LLMDesignResult:
    """完整流程：发 prompt → 解析 → 校验 →（失败回退默认算法）。

    返回 LLMDesignResult；spec 永远非 None（失败时是默认算法，error 说明原因）。
    """
    encoding = "permutation" if problem.name != "knapsack" else "binary"
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(
            problem, instance, n_search_steps, default_iterations)},
    ]
    raw = LLMClient.retry_chat(client, messages, temperature=temperature)

    try:
        data = _extract_json(raw)
        spec, err = spec_from_llm_json(
            data, problem.name, encoding, n_search_steps, default_iterations)
        if err:
            return LLMDesignResult(default_spec(problem.name, encoding, default_iterations),
                                   raw, error=err, used_fallback=True, parsed=data)
        return LLMDesignResult(spec, raw, parsed=data)
    except ValueError as e:
        return LLMDesignResult(default_spec(problem.name, encoding, default_iterations),
                               raw, error=str(e), used_fallback=True)
