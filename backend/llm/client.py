"""
阿里云百炼 LLM 客户端模块
通过 DashScope 兼容接口（OpenAI 兼容模式）调用大模型
支持对话生成和文本嵌入两种能力
"""
import logging
from typing import Iterator
from openai import OpenAI

from backend.config.settings import get_settings

logger = logging.getLogger("ai_rd_agent")


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
        self._client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_URL,
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
            response = self._client.chat.completions.create(
                model=self._chat_model,
                messages=messages,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                stream=stream,
            )

            if stream:
                # 流式模式：返回一个生成器，逐 token 产出
                return self._stream_response(response)
            else:
                # 非流式模式：直接返回完整文本
                return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"LLM 对话调用失败: {e}", exc_info=True)
            raise

    def chat_simple(self, user_message: str, system_prompt: str = "") -> str:
        """简化的对话接口 — 只需传入用户消息和可选的系统提示

        Args:
            user_message: 用户输入文本
            system_prompt: 系统角色设定（可选）

        Returns:
            模型回复字符串
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return self.chat(messages)

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
            # 批量调用嵌入 API
            response = self._client.embeddings.create(
                model=self._embed_model,
                input=texts,
            )
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
