"""
OpenAI 官方 Provider
"""
from backend.config.settings import get_settings
from backend.llm.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI 官方 API"""

    name = "openai"

    def __init__(self):
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY 未配置！请在 .env 文件中设置 OpenAI API Key"
            )
        super().__init__(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_URL,
            chat_model=settings.OPENAI_MODEL,
            embed_model=settings.OPENAI_EMBED_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
