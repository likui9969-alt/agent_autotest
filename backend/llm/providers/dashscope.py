"""
阿里云百炼 DashScope Provider
"""
from backend.config.settings import get_settings
from backend.llm.providers.openai_compatible import OpenAICompatibleProvider


class DashScopeProvider(OpenAICompatibleProvider):
    """阿里云百炼 / DashScope"""

    name = "dashscope"

    def __init__(self):
        settings = get_settings()
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError(
                "DASHSCOPE_API_KEY 未配置！请在 .env 文件中设置阿里云百炼 API Key"
            )
        super().__init__(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_URL,
            chat_model=settings.LLM_MODEL,
            embed_model=settings.EMBEDDING_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
