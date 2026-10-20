"""算法构件库：导入时自动注册全部构件（初始化/搜索/接受三类）。"""

from . import accept_rules, init_ops, search_ops  # noqa: F401  触发构件注册
from .registry import AlgorithmSpec, Component, describe, get, list_for, register

__all__ = ["AlgorithmSpec", "Component", "describe", "get", "list_for", "register"]
