"""OpenAI 兼容的大模型客户端（支持 DeepSeek / OpenAI / 智谱 GLM / 通义千问等）。

只需切换 base_url 与 model 即可接入不同服务商：
- DeepSeek:   https://api.deepseek.com         模型: deepseek-chat
- OpenAI:     https://api.openai.com/v1        模型: gpt-4o-mini
- 智谱 GLM:   https://open.bigmodel.cn/api/paas/v4  模型: glm-4-flash
- 通义千问:   https://dashscope.aliyuncs.com/compatible-mode/v1  模型: qwen-plus

API Key 通过环境变量 `LLM_API_KEY`（或构造参数）提供，不硬编码在代码中。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import requests

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 90.0


def load_config(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMConfig:
    """从环境变量与参数合并读取配置（环境变量优先作为兜底）。"""
    return LLMConfig(
        api_key=api_key or os.environ.get("LLM_API_KEY", ""),
        base_url=base_url or os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        model=model or os.environ.get("LLM_MODEL", DEFAULT_MODEL),
    )


class LLMClient:
    """极简 OpenAI 兼容 chat/completions 客户端（requests 实现）。"""

    def __init__(self, config: LLMConfig):
        self.config = config

    def is_configured(self) -> bool:
        return bool(self.config.api_key.strip())

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        """发送对话，返回助手回复文本。"""
        if not self.is_configured():
            raise RuntimeError(
                "未配置大模型 API Key。请设置环境变量 LLM_API_KEY，"
                "或用 --key 参数传入；可选 LLM_BASE_URL / LLM_MODEL 切换服务商。"
            )
        cfg = self.config
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"请求大模型 API 失败（网络/连接）：{e}") from e

        if resp.status_code != 200:
            detail = resp.text[:300]
            if resp.status_code == 401:
                raise RuntimeError("大模型 API Key 无效或未授权（401）。")
            if resp.status_code == 429:
                raise RuntimeError("大模型 API 请求过于频繁（429），请稍后重试。")
            raise RuntimeError(f"大模型 API 返回 {resp.status_code}：{detail}")

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"大模型 API 返回格式异常：{e}") from e

    @staticmethod
    def retry_chat(client: "LLMClient", messages: list[dict], retries: int = 2, **kw) -> str:
        """带退避的调用（对 429/瞬时错误重试）。"""
        for attempt in range(retries + 1):
            try:
                return client.chat(messages, **kw)
            except RuntimeError as e:
                if attempt < retries and "429" in str(e):
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("重试后仍失败")
