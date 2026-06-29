"""
LLM 供应商抽象层
定义统一的 LLM Provider 接口，支持多供应商实现和自动回退。
参考：Ai_Test_Agent-main 的 BaseLLMProvider 设计

供应商适配模式：
  1. 定义 LLMProvider 抽象基类（chat / embed / chat_with_tools）
  2. DashScopeProvider：现有实现（直接使用 OpenAI 兼容客户端）
  3. OllamaProvider：本地 fallback，通过 Ollama 兼容端点
  4. LLMProviderFactory：根据配置初始化 provider 链，自动 fallback
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("ai_rd_agent")


# ==================== 数据类 ====================

@dataclass
class ProviderConfig:
    """单个供应商的配置"""
    name: str                     # dashscope / ollama / ...
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    embed_model: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 60.0


@dataclass
class ProviderResponse:
    """统一的 LLM 响应格式"""
    content: str
    tool_calls: list[dict] | None = None  # [{id, name, args}, ...]
    model: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ==================== 抽象基类 ====================

class LLMProvider(ABC):
    """LLM 供应商抽象基类"""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._initialized = False

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """供应商名称（dashscope / ollama）"""
        ...

    @abstractmethod
    def _initialize(self):
        """初始化底层客户端（延迟初始化）"""
        ...

    def ensure_initialized(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """对话生成"""
        ...

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        """带工具调用的对话"""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """文本嵌入"""
        ...

    def embed_single(self, text: str) -> list[float]:
        results = self.embed([text])
        return results[0] if results else []


# ==================== DashScope Provider ====================

class DashScopeProvider(LLMProvider):
    """阿里云百炼 DashScope（OpenAI 兼容模式）"""

    @property
    def provider_name(self) -> str:
        return "dashscope"

    def _initialize(self):
        import httpx
        from openai import OpenAI

        self._http_client = httpx.Client(
            trust_env=False,
            timeout=self.config.timeout,
        )
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            http_client=self._http_client,
        )
        logger.info(
            f"[DashScope] 已初始化: model={self.config.model}, "
            f"embed={self.config.embed_model}"
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        self.ensure_initialized()
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )
        usage = response.usage
        return ProviderResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            finish_reason=response.choices[0].finish_reason or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        import json

        self.ensure_initialized()
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )

        msg = response.choices[0].message
        tool_calls = None
        if msg.tool_calls:
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

        usage = response.usage
        return ProviderResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            model=response.model,
            finish_reason=response.choices[0].finish_reason or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.ensure_initialized()
        response = self._client.embeddings.create(
            model=self.config.embed_model,
            input=texts,
        )
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]


# ==================== Ollama Provider（本地 fallback） ====================

class OllamaProvider(LLMProvider):
    """Ollama 本地模型（OpenAI 兼容模式）"""

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _initialize(self):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.config.api_key or "ollama",
            base_url=self.config.base_url,
        )
        logger.info(f"[Ollama] 已初始化: url={self.config.base_url}, model={self.config.model}")

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        self.ensure_initialized()
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )
        usage = response.usage
        return ProviderResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            finish_reason=response.choices[0].finish_reason or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ProviderResponse:
        # Ollama 工具调用支持有限，走普通 chat + prompt 引导
        logger.warning("[Ollama] chat_with_tools 回退到普通 chat")
        return self.chat(messages, temperature, max_tokens)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.ensure_initialized()
        response = self._client.embeddings.create(
            model=self.config.embed_model or "nomic-embed-text",
            input=texts,
        )
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]


# ==================== 供应商工厂 ====================

class LLMProviderFactory:
    """LLM 供应商工厂 — 管理 provider 链和自动 fallback

    provider 链：['dashscope', 'ollama']
    主 provider 失败后自动切换到下一个 provider（按配置顺序）。
    """

    def __init__(self, providers: list[LLMProvider]):
        """
        Args:
            providers: 按优先级排序的 provider 列表（provider[0] 为主）
        """
        self._providers = providers
        self._current_index = 0
        self._failed_providers: set[int] = set()

    @property
    def current(self) -> LLMProvider:
        """获取当前活跃的 provider"""
        return self._providers[self._current_index]

    @property
    def active_provider_name(self) -> str:
        return self.current.provider_name

    def _try_next(self) -> bool:
        """切换到下一个可用 provider，返回是否切换成功"""
        for i in range(self._current_index + 1, len(self._providers)):
            if i not in self._failed_providers:
                logger.warning(
                    f"[ProviderFactory] 从 {self.current.provider_name} "
                    f"回退到 {self._providers[i].provider_name}"
                )
                self._current_index = i
                return True
        return False

    def execute(self, method: str, *args, **kwargs) -> Any:
        """在 provider 链上执行方法，自动 fallback

        Args:
            method: 方法名（chat / embed / chat_with_tools）
            *args, **kwargs: 方法参数

        Returns:
            方法返回值

        Raises:
            RuntimeError: 所有 provider 都失败时
        """
        last_error = None
        attempted = set()

        while self._current_index not in attempted:
            attempted.add(self._current_index)
            provider = self._providers[self._current_index]

            try:
                fn = getattr(provider, method)
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.error(
                    f"[ProviderFactory] {provider.provider_name}.{method} 失败: {e}"
                )
                self._failed_providers.add(self._current_index)

                if not self._try_next():
                    break

        raise RuntimeError(
            f"所有 LLM provider 均失败 (已尝试: "
            f"{', '.join(self._providers[i].provider_name for i in attempted)}): "
            f"{last_error}"
        ) from last_error

    def chat(self, messages, **kwargs) -> ProviderResponse:
        return self.execute("chat", messages, **kwargs)

    def chat_with_tools(self, messages, tools, **kwargs) -> ProviderResponse:
        return self.execute("chat_with_tools", messages, tools, **kwargs)

    def embed(self, texts, **kwargs) -> list[list[float]]:
        return self.execute("embed", texts, **kwargs)

    def embed_single(self, text: str) -> list[float]:
        results = self.execute("embed", [text])
        return results[0] if results else []


def create_provider_chain(settings) -> list[LLMProvider]:
    """根据配置创建 provider 链

    从 settings.LLM_PROVIDERS 列表和 LLM_FALLBACK_ENABLED 决定 provider 链。

    Returns:
        按优先级排序的 LLMProvider 列表
    """
    providers: list[LLMProvider] = []

    # 主 provider: DashScope
    if settings.DASHSCOPE_API_KEY:
        providers.append(DashScopeProvider(ProviderConfig(
            name="dashscope",
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_URL,
            model=settings.LLM_MODEL,
            embed_model=settings.EMBEDDING_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )))

    # Fallback: Ollama（如启用且配置了 URL）
    if settings.LLM_FALLBACK_ENABLED and settings.OLLAMA_URL:
        providers.append(OllamaProvider(ProviderConfig(
            name="ollama",
            base_url=settings.OLLAMA_URL,
            model=settings.OLLAMA_MODEL or "qwen2.5:7b",
            embed_model=settings.OLLAMA_EMBED_MODEL or "nomic-embed-text",
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )))

    if not providers:
        raise ValueError(
            "未配置任何 LLM provider。请设置 DASHSCOPE_API_KEY 或 "
            "启用 LLM_FALLBACK_ENABLED 并配置 OLLAMA_URL"
        )

    logger.info(
        f"[ProviderFactory] provider 链: "
        f"{' → '.join(p.provider_name for p in providers)}"
    )
    return providers
