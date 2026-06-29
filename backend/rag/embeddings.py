"""
嵌入生成模块
使用阿里云百炼的 Embedding 模型将文本转换为向量
"""
import logging
from typing import Iterator
from backend.llm.client import LLMClient

logger = logging.getLogger("ai_rd_agent")

# 单次嵌入 API 调用的最大文本数
# DashScope text-embedding-v3 限制单次最多 2048 条输入，
# 设为 512 以平衡吞吐与容错（一批失败只丢 512 条）
_EMBED_BATCH_SIZE = 512


def _batch_iter(texts: list[str], batch_size: int) -> Iterator[list[str]]:
    """将文本列表分批

    Args:
        texts: 全部文本
        batch_size: 每批大小

    Yields:
        每批文本子列表
    """
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


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

        超过 _EMBED_BATCH_SIZE 条文本时自动分批调用 API，
        按原始顺序拼接返回。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表（与输入顺序一致）
        """
        if not texts:
            return []

        logger.info(f"正在生成嵌入向量: {len(texts)} 个文本...")

        all_vectors: list[list[float]] = []
        total_batches = (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE

        for batch_idx, batch in enumerate(_batch_iter(texts, _EMBED_BATCH_SIZE)):
            if total_batches > 1:
                logger.info(
                    f"  嵌入批次 {batch_idx + 1}/{total_batches} "
                    f"({len(batch)} 条文本)..."
                )
            vectors = self._client.embed(batch)
            all_vectors.extend(vectors)

        logger.info(f"嵌入生成完成: {len(all_vectors)} 个向量 ({total_batches} 批次)")
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """对单条查询文本进行嵌入

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        return self._client.embed_single(text)
