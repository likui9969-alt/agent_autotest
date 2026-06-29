"""
OpenAI 兼容接口 Provider 基类

DashScope、Ollama、OpenAI 原生均兼容 OpenAI SDK，
统一在此基类实现，子类只需提供 api_key、base_url、model。
"""
import logging
from typing import Iterator

import httpx
from openai import OpenAI

from backend.llm.providers.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger("ai_rd_agent")


class OpenAICompatibleProvider(BaseLLMProvider):
    """基于 OpenAI SDK 的通用 Provider"""

    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        chat_model: str,
        embed_model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._chat_model = chat_model
        self._embed_model = embed_model
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._http_client = httpx.Client(trust_env=False, timeout=timeout)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self._http_client,
        )
        logger.info(
            f"[{self.name}] Provider 初始化 | 对话模型: {chat_model} | "
            f"嵌入模型: {embed_model} | 端点: {base_url}"
        )

    def is_available(self) -> bool:
        return bool(self._api_key and self._base_url and self._chat_model)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        response = self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            stream=stream,
        )
        if stream:
            return self._stream_response(response)
        return response.choices[0].message.content or ""

    def _stream_response(self, response) -> Iterator[str]:
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
        )

        msg = response.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            import json
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                })

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            usage=usage,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._embed_model,
            input=texts,
        )
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]
