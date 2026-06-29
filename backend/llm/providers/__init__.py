"""
LLM Provider 包

提供多 LLM 后端统一接口，支持 DashScope / OpenAI / Ollama 等。
"""
from backend.llm.providers.base import BaseLLMProvider, LLMResponse
from backend.llm.providers.factory import create_provider

__all__ = ["BaseLLMProvider", "LLMResponse", "create_provider"]
