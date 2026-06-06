"""
嵌入生成模块
使用阿里云百炼的 Embedding 模型将文本转换为向量
"""
import logging
from backend.llm.client import LLMClient

logger = logging.getLogger("ai_rd_agent")


class EmbeddingGenerator:
    """文本嵌入生成器

    封装 LLMClient 的 embed 方法，提供面向 RAG 场景的便捷接口。

    使用示例：
        gen = EmbeddingGenerator()
        vectors = gen.embed_documents(["文档1文本", "文档2文本"])
        vector = gen.embed_query("用户查询文本")
    """

    def __init__(self, llm_client: LLMClient | None = None):
        """
        Args:
            llm_client: LLM 客户端实例（不传则自动创建）
        """
        self._client = llm_client or LLMClient()
        logger.info("嵌入生成器已初始化")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """对多个文档文本进行批量嵌入

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        logger.info(f"正在生成嵌入向量: {len(texts)} 个文本...")
        vectors = self._client.embed(texts)
        logger.info(f"嵌入生成完成: {len(vectors)} 个向量")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """对单条查询文本进行嵌入

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        return self._client.embed_single(text)
