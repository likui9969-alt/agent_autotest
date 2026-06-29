"""
Ollama 本地模型 Provider

Ollama 提供 OpenAI 兼容接口，地址通常为 http://localhost:11434/v1
"""
from backend.config.settings import get_settings
from backend.llm.providers.openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama 本地模型"""

    name = "ollama"

    def __init__(self):
        settings = get_settings()
        if not settings.OLLAMA_URL:
            raise ValueError(
                "OLLAMA_URL 未配置！请在 .env 文件中设置 Ollama 服务地址，"
                "例如 http://localhost:11434/v1"
            )
        super().__init__(
            api_key="ollama",  # Ollama 不需要 api_key，但 OpenAI 客户端要求非空
            base_url=settings.OLLAMA_URL,
            chat_model=settings.OLLAMA_MODEL,
            embed_model=settings.OLLAMA_EMBED_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
