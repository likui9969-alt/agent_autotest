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
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    APIStatusError,
)

from backend.config.settings import get_settings
from backend.llm.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitBreakerConfig
from backend.llm.providers import create_provider, BaseLLMProvider

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
    """多 Provider LLM 客户端

    支持 DashScope / OpenAI / Ollama 等多种后端，按配置自动切换：
    1. chat()  — 对话生成（用于 RAG 问答、故障分析等）
    2. embed() — 文本嵌入（用于文档向量化）
    3. chat_with_tools() — 带工具调用的对话

    调用示例：
        client = LLMClient()
        response = client.chat("你好，请介绍一下自己")
        embeddings = client.embed(["这是一段需要向量化的文本"])
    """

    def __init__(self):
        """初始化客户端 — 从全局配置读取 Provider 列表"""
        settings = get_settings()

        # provider 优先级：LLM_PROVIDERS > LLM_PROVIDER
        provider_names = settings.LLM_PROVIDERS
        if not provider_names:
            provider_names = [settings.LLM_PROVIDER]

        self._providers: list[BaseLLMProvider] = []
        for name in provider_names:
            try:
                provider = create_provider(name)
                if provider.is_available():
                    self._providers.append(provider)
                    logger.info(f"LLM 已注册 provider: {name}")
            except Exception as e:
                logger.warning(f"LLM provider {name} 初始化失败: {e}")

        if not self._providers:
            raise ValueError(
                "没有可用的 LLM Provider，请检查 .env 配置中的 "
                "DASHSCOPE_API_KEY / OPENAI_API_KEY / OLLAMA_URL"
            )

        self._current_index = 0
        self._chat_model = settings.LLM_MODEL
        self._embed_model = settings.EMBEDDING_MODEL

        # 初始化熔断器（从配置读取参数）
        self._breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=settings.LLM_CIRCUIT_BREAKER_THRESHOLD,
            recovery_timeout=settings.LLM_CIRCUIT_BREAKER_RECOVERY,
            success_threshold=settings.LLM_CIRCUIT_BREAKER_SUCCESS,
        ))

    @property
    def _current_provider(self) -> BaseLLMProvider:
        return self._providers[self._current_index]

    def _next_provider(self) -> None:
        """切换到下一个 Provider（循环）"""
        if len(self._providers) > 1:
            self._current_index = (self._current_index + 1) % len(self._providers)
            logger.info(
                f"LLM 切换 Provider: {self._current_provider.name}"
            )

    # ==================== 重试机制 ====================

    def _call_with_retry(self, provider: BaseLLMProvider, fn: Callable[[BaseLLMProvider], T]) -> T:
        """以指数退避重试执行单个 Provider 的 LLM 调用，带熔断器保护

        熔断器开启时快速失败（不发起网络请求）。
        成功/失败后通知熔断器更新状态。
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
                result = fn(provider)
                self._breaker.record_success()
                return result
            except CircuitBreakerOpenError:
                raise
            except Exception as e:
                last_exc = e
                self._breaker.record_failure()
                if not _is_retryable(e):
                    raise

                if attempt == max_attempts:
                    logger.error(f"LLM 调用重试 {max_attempts} 次均失败: {e}")
                    raise

                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                jitter = random.uniform(-0.5, 0.5)
                sleep_time = max(0.1, delay + jitter)

                logger.warning(
                    f"[{provider.name}] LLM 调用失败 (尝试 {attempt}/{max_attempts})，"
                    f"{sleep_time:.1f}s 后重试: {type(e).__name__}: {e}"
                )
                time.sleep(sleep_time)

        raise last_exc  # type: ignore[misc]

    def _call(self, fn: Callable[[BaseLLMProvider], T]) -> T:
        """跨 Provider 调用：当前 Provider 失败后自动切换到下一个"""
        last_exc = None
        for _ in range(len(self._providers)):
            provider = self._current_provider
            try:
                return self._call_with_retry(provider, fn)
            except CircuitBreakerOpenError:
                raise
            except Exception as e:
                last_exc = e
                logger.warning(f"[{provider.name}] 调用失败，尝试切换 Provider: {e}")
                self._next_provider()

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
            if stream:
                # 流式模式下不做跨 Provider 切换（generator 延迟执行）
                return self._current_provider.chat(
                    messages, temperature, max_tokens, stream=True
                )
            return self._call(
                lambda p: p.chat(messages, temperature, max_tokens, stream=False)
            )
        except Exception as e:
            logger.error(f"LLM 对话调用失败: {e}", exc_info=True)
            raise

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
            return self._call(lambda p: p.embed(texts))
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
            {"content": str, "tool_calls": list | None, "usage": dict | None}
        """
        try:
            response = self._call(
                lambda p: p.chat_with_tools(
                    messages, tools, tool_choice, temperature, max_tokens
                )
            )
            return {
                "content": response.content,
                "tool_calls": response.tool_calls or None,
                "usage": response.usage,
            }
        except Exception as e:
            logger.error(f"LLM 工具调用失败: {e}", exc_info=True)
            raise
