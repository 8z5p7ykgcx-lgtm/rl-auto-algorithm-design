"""LLM 算法设计器测试：响应解析、校验、回退与端到端（mock，不调真实 API）。"""

import json

import numpy as np
import pytest

from auto_alg_design import TSP
from auto_alg_design.llm.designer import (
    _extract_json,
    build_user_prompt,
    default_spec,
    design_with_llm,
    spec_from_llm_json,
)
from auto_alg_design.llm.client import load_config
from auto_alg_design.components.runner import run_algorithm


class MockChat:
    """可编程的 mock 客户端，绕过网络直接返回预设文本。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def is_configured(self) -> bool:
        return True

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.reply


def _payload(action="compose", algo=None):
    return json.dumps({"action": action, "reasoning": "test", "algorithm": algo}, ensure_ascii=False)


def test_extract_json_inside_fence():
    text = '好的，这是设计：\n```json\n{"action": "compose", "algorithm": {"init": {"id": "x", "param": 0}}}\n```\n完毕'
    assert _extract_json(text)["action"] == "compose"


def test_parse_valid_compose_tsp():
    data = json.loads(_payload(algo={
        "init": {"id": "greedy_tsp", "param": 1},
        "search_steps": [
            {"id": "two_opt_ls", "param": 2},
            {"id": "swap_mutation", "param": 0},
        ],
        "accept": {"id": "elite_accept", "param": 0},
        "iterations": 80,
    }))
    spec, err = spec_from_llm_json(data, "tsp", "permutation", max_search_steps=4)
    assert err == "" and spec is not None
    assert spec.init_id == "greedy_tsp" and spec.init_param == 1
    assert [s[0] for s in spec.search_steps] == ["two_opt_ls", "swap_mutation"]
    assert spec.accept_id == "elite_accept" and spec.iterations == 80


def test_parse_rejects_unknown_component():
    data = json.loads(_payload(algo={
        "init": {"id": "no_such_init", "param": 0},
        "search_steps": [{"id": "two_opt_ls", "param": 0}],
        "accept": {"id": "elite_accept", "param": 0},
    }))
    spec, err = spec_from_llm_json(data, "tsp", "permutation")
    assert spec is None and "no_such_init" in err


def test_parse_rejects_param_out_of_range():
    data = json.loads(_payload(algo={
        "init": {"id": "greedy_tsp", "param": 0},
        "search_steps": [{"id": "two_opt_ls", "param": 9}],
        "accept": {"id": "elite_accept", "param": 0},
    }))
    spec, err = spec_from_llm_json(data, "tsp", "permutation")
    assert spec is None and "param" in err


def test_parse_rejects_wrong_problem_component():
    # flowshop 上不允许用 two_opt_ls（TSP 专属）
    data = json.loads(_payload(algo={
        "init": {"id": "neh_init", "param": 0},
        "search_steps": [{"id": "two_opt_ls", "param": 0}],
        "accept": {"id": "elite_accept", "param": 0},
    }))
    spec, err = spec_from_llm_json(data, "flowshop", "permutation")
    assert spec is None and "two_opt_ls" in err


def test_parse_select_action():
    data = json.loads(_payload(action="select", algo="ga"))
    spec, err = spec_from_llm_json(data, "tsp", "permutation")
    assert err == "" and spec.meta.get("select") == "ga"
    data_bad = json.loads(_payload(action="select", algo="xxx"))
    spec, err = spec_from_llm_json(data_bad, "tsp", "permutation")
    assert spec is None and "xxx" in err


def test_parse_missing_algorithm():
    spec, err = spec_from_llm_json({"action": "compose"}, "tsp", "permutation")
    assert spec is None and "algorithm" in err


def test_design_with_llm_end_to_end():
    p = TSP(n=20, seed=1)
    inst = p.random_instance(np.random.default_rng(1))
    client = MockChat(
        _payload(algo={
            "init": {"id": "greedy_tsp", "param": 0},
            "search_steps": [{"id": "two_opt_ls", "param": 1}, {"id": "swap_mutation", "param": 0}],
            "accept": {"id": "elite_accept", "param": 0},
            "iterations": 40,
        })
    )
    result = design_with_llm(client, p, inst, n_search_steps=4, default_iterations=40)
    assert result.error == "" and not result.used_fallback
    assert result.spec.meta.get("source") == "llm"
    # 设计出的算法真正可运行
    cost, traj, sol = run_algorithm(p, inst, result.spec, np.random.default_rng(2), record_trajectory=True)
    assert len(traj) == 40 and cost > 0
    # prompt 里应包含构件清单
    user_msg = client.calls[0][1]["content"]
    assert "greedy_tsp" in user_msg and "two_opt_ls" in user_msg


def test_design_with_llm_fallback_on_garbage():
    p = TSP(n=20, seed=1)
    inst = p.random_instance(np.random.default_rng(1))
    client = MockChat("抱歉，我无法给出 JSON。")
    result = design_with_llm(client, p, inst)
    assert result.used_fallback is True and result.error != ""
    assert result.spec.meta.get("source") == "fallback"


def test_build_user_prompt_contains_features():
    p = TSP(n=20, seed=1)
    inst = p.random_instance(np.random.default_rng(1))
    prompt = build_user_prompt(p, inst)
    assert "tsp" in prompt and "n_cities" in prompt and "dist_mean" in prompt
    assert "档位范围" in prompt and "greedy_tsp" in prompt


def test_default_spec_runnable_knapsack():
    from auto_alg_design import Knapsack
    p = Knapsack(n=30, seed=2)
    inst = p.random_instance(np.random.default_rng(2))
    spec = default_spec("knapsack", "binary")
    cost, _, sol = run_algorithm(p, inst, spec, np.random.default_rng(3))
    assert sol @ inst["weights"] <= inst["capacity"] + 1e-9


def test_load_config_env():
    import os
    os.environ["LLM_API_KEY"] = "sk-test"
    os.environ["LLM_MODEL"] = "deepseek-chat"
    cfg = load_config()
    assert cfg.api_key == "sk-test" and cfg.model == "deepseek-chat"
    os.environ.pop("LLM_API_KEY", None)
    os.environ.pop("LLM_MODEL", None)
    cfg2 = load_config()
    assert cfg2.api_key == ""  # 清除后回到默认
    assert cfg2.model == "deepseek-chat"
