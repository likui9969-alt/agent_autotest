"""
阿里云百炼 LLM 客户端模块
通过 DashScope 兼容接口（OpenAI 兼容模式）调用大模型
支持对话生成和文本嵌入两种能力
"""
import logging
import random
import time
from typing import Iterator, Callable, TypeVar

import httpx
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    APIStatusError,
)

from backend.config.settings import get_settings
from backend.llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitBreakerConfig

T = TypeVar("T")
logger = logging.getLogger("ai_rd_agent")

# 不可重试的 HTTP 状态码
_NON_RETRYABLE_STATUSES = frozenset({400, 401, 403, 404, 413, 422})


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否值得重试"""
    # OpenAI 客户端明确的可重试异常
    if isinstance(exc, (APITimeoutError, APIConnectionError, InternalServerError)):
        return True
    if isinstance(exc, RateLimitError):
        return True  # 429 限流，退避后重试
    # APIStatusError — 只在状态码不在不可重试集合时重试
    if isinstance(exc, APIStatusError):
        return exc.status_code not in _NON_RETRYABLE_STATUSES
    # httpx 传输层错误
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)):
        return True
    return False


class LLMClient:
    """阿里云百炼 LLM 客户端

    使用 DashScope 的 OpenAI 兼容接口，同时支持：
    1. chat()  — 对话生成（用于 RAG 问答、故障分析等）
    2. embed() — 文本嵌入（用于文档向量化）

    调用示例：
        client = LLMClient()
        response = client.chat("你好，请介绍一下自己")
        embeddings = client.embed(["这是一段需要向量化的文本"])
    """

    def __init__(self):
        """初始化客户端 — 从全局配置读取 API Key 和 Base URL"""
        settings = get_settings()

        if not settings.DASHSCOPE_API_KEY:
            raise ValueError(
                "DASHSCOPE_API_KEY 未配置！请在 .env 文件中设置阿里云百炼 API Key"
            )

        # 使用 OpenAI 兼容客户端指向 DashScope 端点
        # 显式构造 httpx.Client 并禁用环境代理（trust_env=False），
        # 避免某些启动环境注入的 HTTP_PROXY/HTTPS_PROXY 导致连接失败。
        self._http_client = httpx.Client(trust_env=False, timeout=60.0)
        self._client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_URL,
            http_client=self._http_client,
        )
        self._chat_model = settings.LLM_MODEL
        self._embed_model = settings.EMBEDDING_MODEL
        self._temperature = settings.LLM_TEMPERATURE
        self._max_tokens = settings.LLM_MAX_TOKENS

        logger.info(
            f"LLM 客户端已初始化 | "
            f"对话模型: {self._chat_model} | "
            f"嵌入模型: {self._embed_model} | "
            f"端点: {settings.DASHSCOPE_URL}"
        )

        # 初始化熔断器（从配置读取参数）
        self._breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=settings.LLM_CIRCUIT_BREAKER_THRESHOLD,
            recovery_timeout=settings.LLM_CIRCUIT_BREAKER_RECOVERY,
            success_threshold=settings.LLM_CIRCUIT_BREAKER_SUCCESS,
        ))

    # ==================== 重试机制 ====================

    def _call_with_retry(self, fn: Callable[[], T]) -> T:
        """以指数退避重试执行 LLM 调用，带熔断器保护

        熔断器开启时快速失败（不发起网络请求）。
        成功/失败后通知熔断器更新状态。

        Args:
            fn: 无参可调用对象，封装实际的 LLM API 调用

        Returns:
            fn 的返回值

        Raises:
            CircuitBreakerOpenError: 熔断器开启，请求被拒绝
            最后一次尝试的异常（所有重试均失败后）
        """
        # 熔断器检查
        if self._breaker.is_open:
            logger.warning("LLM 调用被熔断器拒绝（快速失败）")
            raise CircuitBreakerOpenError("熔断器开启，LLM 调用被拒绝")

        settings = get_settings()
        max_attempts = settings.LLM_RETRY_MAX_ATTEMPTS
        base_delay = settings.LLM_RETRY_BASE_DELAY
        max_delay = settings.LLM_RETRY_MAX_DELAY

        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = fn()
                self._breaker.record_success()
                return result
            except CircuitBreakerOpenError:
                raise  # 熔断器拒绝，立即抛出
            except Exception as e:
                last_exc = e
                self._breaker.record_failure()
                if not _is_retryable(e):
                    raise  # 不可重试，立即抛出

                if attempt == max_attempts:
                    logger.error(
                        f"LLM 调用重试 {max_attempts} 次均失败: {e}"
                    )
                    raise  # 最后一次尝试仍失败，抛出

                # 指数退避 + 抖动
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                jitter = random.uniform(-0.5, 0.5)
                sleep_time = max(0.1, delay + jitter)

                logger.warning(
                    f"LLM 调用失败 (尝试 {attempt}/{max_attempts})，"
                    f"{sleep_time:.1f}s 后重试: {type(e).__name__}: {e}"
                )
                time.sleep(sleep_time)

        raise last_exc  # type: ignore[misc]

    # ==================== 对话生成 ====================

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """发送对话请求，返回模型回复

        Args:
            messages: 消息列表，格式 [{"role": "system/user/assistant", "content": "..."}]
            temperature: 生成温度（0~2），越低越确定。不传则使用全局配置
            max_tokens: 最大输出 token 数。不传则使用全局配置
            stream: 是否流式输出

        Returns:
            非流式：返回完整回复字符串
            流式：返回逐 token 迭代器
        """
        try:
            response = self._call_with_retry(lambda: (
                self._client.chat.completions.create(
                    model=self._chat_model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self._temperature,
                    max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                    stream=stream,
                )
            ))

            if stream:
                # 流式模式：返回一个生成器，逐 token 产出
                return self._stream_response(response)
            else:
                # 非流式模式：直接返回完整文本
                return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"LLM 对话调用失败: {e}", exc_info=True)
            raise

    def _stream_response(self, response) -> Iterator[str]:
        """处理流式响应，逐 token 产出"""
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ==================== 文本嵌入 ====================

    def embed(self, texts: list[str]) -> list[list[float]]:
        """对文本列表进行向量嵌入

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表，每个向量是 float 列表
        """
        if not texts:
            return []

        try:
            response = self._call_with_retry(lambda: (
                self._client.embeddings.create(
                    model=self._embed_model,
                    input=texts,
                )
            ))
            # 按输入顺序返回向量
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in embeddings]

        except Exception as e:
            logger.error(f"LLM 嵌入调用失败: {e}", exc_info=True)
            raise

    def embed_single(self, text: str) -> list[float]:
        """对单条文本进行向量嵌入

        Args:
            text: 待嵌入的文本

        Returns:
            嵌入向量
        """
        results = self.embed([text])
        return results[0] if results else []

    # ==================== 带工具调用的对话 ====================

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """发送带工具定义的对话请求，返回 Choice 消息对象

        Args:
            messages: 消息列表
            tools: OpenAI function calling 工具定义
            tool_choice: 工具选择策略 (auto/none/required)
            temperature: 生成温度
            max_tokens: 最大输出 token 数

        Returns:
            {"content": str, "tool_calls": list | None}
        """
        try:
            response = self._call_with_retry(lambda: (
                self._client.chat.completions.create(
                    model=self._chat_model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature if temperature is not None else self._temperature,
                    max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                )
            ))

            msg = response.choices[0].message
            tool_calls = None
            if msg.tool_calls:
                import json
                tool_calls = []
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

            return {
                "content": msg.content or "",
                "tool_calls": tool_calls,
            }

        except Exception as e:
            logger.error(f"LLM 工具调用失败: {e}", exc_info=True)
            raise
