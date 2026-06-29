"""
LLM Provider 工厂

根据 provider 名称创建对应 Provider 实例，支持自动回退。
"""
import logging

from backend.llm.providers.base import BaseLLMProvider
from backend.llm.providers.dashscope import DashScopeProvider
from backend.llm.providers.openai import OpenAIProvider
from backend.llm.providers.ollama import OllamaProvider

logger = logging.getLogger("ai_rd_agent")

_PROVIDER_MAP: dict[str, type[BaseLLMProvider]] = {
    "dashscope": DashScopeProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def create_provider(name: str) -> BaseLLMProvider:
    """根据名称创建 Provider 实例"""
    provider_cls = _PROVIDER_MAP.get(name.lower())
    if provider_cls is None:
        raise ValueError(
            f"未知 LLM Provider: {name}。支持的 provider: {list(_PROVIDER_MAP.keys())}"
        )
    return provider_cls()


def list_providers() -> list[str]:
    """返回支持的 provider 名称列表"""
    return list(_PROVIDER_MAP.keys())
