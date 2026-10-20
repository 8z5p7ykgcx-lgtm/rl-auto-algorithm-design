"""大模型（LLM）驱动的自动算法设计。

思路：用户按模板提供问题实例数据 → 本模块把实例统计特征 + 构件库清单发送给
DeepSeek 等大模型 → 大模型输出「算法设计决策」（构件组合 + 参数档位，或直接
选择现成算法）→ 校验后交给 runner 执行求解。对应综述中「大模型赋能自动算法
设计」的方向。
"""

from .client import LLMClient, load_config
from .designer import (
    build_system_prompt,
    build_user_prompt,
    design_with_llm,
    spec_from_llm_json,
)

__all__ = [
    "LLMClient", "load_config",
    "build_system_prompt", "build_user_prompt",
    "design_with_llm", "spec_from_llm_json",
]
