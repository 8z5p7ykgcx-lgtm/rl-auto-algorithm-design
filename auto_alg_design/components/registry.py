"""算法构件注册表。

构件 = 可复用的最小算法模块（对应综述 GSF 框架中的「初始化构件、搜索构件、
更新/接受构件」）。RL 智能体的任务就是**从构件库中挑选并组装**出一个完整算法，
这正是方向1「算法构件自动组合」的落点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..problems.base import Problem

# 构件签名约定
InitFn = Callable[[Problem, dict[str, Any], np.random.Generator, int], np.ndarray]
SearchFn = Callable[[Problem, dict[str, Any], np.ndarray, np.random.Generator, int], np.ndarray]
AcceptFn = Callable[[Problem, dict[str, Any], np.ndarray, np.ndarray, np.random.Generator, int], tuple[np.ndarray, Any]]


@dataclass
class Component:
    cid: str                    # 构件 id，如 "swap_mutation"
    kind: str                   # "init" | "search" | "accept"
    encoding: str               # 适用的编码类型："permutation" | "binary" | "any"
    param_values: list[int]     # 离散参数档位（整数索引，映射到具体参数）
    fn: Callable[..., Any]
    desc: str = ""
    problems: set[str] | None = None   # 适用的问题类型（None = 通用）


_REGISTRY: dict[str, Component] = {}


def register(comp: Component) -> Component:
    _REGISTRY[comp.cid] = comp
    return comp


def get(cid: str) -> Component:
    return _REGISTRY[cid]


def list_for(kind: str, encoding: str, problem_name: str | None = None) -> list[Component]:
    """返回某问题可用的构件（init / search / accept 分别取；支持按问题类型过滤）。"""
    out = []
    for c in _REGISTRY.values():
        if c.kind == kind and c.encoding in {encoding, "any"}:
            if c.problems is None or problem_name is None or problem_name in c.problems:
                out.append(c)
    return out


def describe(spec: "AlgorithmSpec") -> str:
    """把自动生成的算法描述成可读字符串（用于展示智能体「设计」出的算法）。"""
    parts = [f"init={spec.init_id}#{spec.init_param}"]
    for cid, p in spec.search_steps:
        parts.append(f"{cid}#{p}")
    parts.append(f"accept={spec.accept_id}#{spec.accept_param}")
    return " -> ".join(parts)


@dataclass
class AlgorithmSpec:
    """一个由 RL 自动设计出的完整算法描述。"""

    init_id: str
    init_param: int                          # 初始化构件参数档位
    search_steps: list[tuple[str, int]]      # [(搜索构件id, 参数档位), ...]
    accept_id: str
    accept_param: int
    iterations: int = 60                     # 主循环迭代次数（可视为算法预算）
    meta: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return describe(self)
