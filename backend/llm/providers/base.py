"""
LLM Provider 抽象基类
"""
from abc import ABC, abstractmethod
from typing import Iterator


class LLMResponse:
    """LLM 调用返回结果"""

    def __init__(
        self,
        content: str = "",
        tool_calls: list[dict] | None = None,
        usage: dict | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage


class BaseLLMProvider(ABC):
    """LLM Provider 抽象接口"""

    name: str = ""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """对话生成"""
        pass

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """带工具调用的对话"""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """文本嵌入"""
        pass

    def is_available(self) -> bool:
        """Provider 是否可用（由子类根据配置判断）"""
        return True
