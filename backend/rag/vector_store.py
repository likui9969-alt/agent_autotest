"""
Chroma 向量存储封装模块
提供文档块的持久化存储、查询和元数据管理
"""
import logging
import uuid
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from backend.config.settings import get_settings

logger = logging.getLogger("ai_rd_agent")

# 默认集合名称
DEFAULT_COLLECTION = "knowledge_base"


class VectorStore:
    """Chroma 向量存储管理器

    负责：
    - 创建和管理 Chroma 集合
    - 文档块的批量写入（含向量和元数据）
    - 集合信息查询和统计

    使用示例：
        store = VectorStore()
        store.add_documents(chunks, embeddings)
        stats = store.get_stats()
    """

    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        """
        Args:
            collection_name: Chroma 集合名称（相当于表名）
        """
        settings = get_settings()
        self._persist_dir = settings.get_chroma_dir()
        self._collection_name = collection_name

        # 确保持久化目录存在
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        # 创建 Chroma 持久化客户端
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # 获取或创建集合（使用余弦相似度）
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={
                "description": "AI研发效能智能体知识库",
                "hnsw:space": "cosine",  # 使用余弦相似度
            },
        )

        logger.info(
            f"Chroma 向量库已连接 | "
            f"集合: {self._collection_name} | "
            f"目录: {self._persist_dir} | "
            f"当前文档数: {self._collection.count()}"
        )

    # ==================== 写入操作 ====================

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        """批量写入文档块到向量库

        Args:
            documents: 切割后的文档块列表
            embeddings: 对应的嵌入向量列表（与 documents 顺序一致）

        Returns:
            写入的块 ID 列表
        """
        if not documents:
            return []

        # 为每个块生成唯一 ID
        ids = [str(uuid.uuid4()) for _ in documents]

        # 提取文本内容和元数据
        texts = [doc.page_content for doc in documents]
        metadatas = [
            {
                **doc.metadata,
                "chunk_index": i,
            }
            for i, doc in enumerate(documents)
        ]

        # 批量写入 Chroma
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"向量库写入完成: {len(ids)} 个块")
        return ids

    # ==================== 查询操作 ====================

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        include_embeddings: bool = False,
    ) -> dict:
        """按向量相似度查询最相关的文档块

        Args:
            query_embedding: 查询文本的嵌入向量
            top_k: 返回结果数量
            include_embeddings: 是否返回嵌入向量（MMR 需要）

        Returns:
            Chroma 原生查询结果（含 ids, documents, metadatas, distances）
        """
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=include,
        )

    def get_all_documents(self) -> dict:
        """获取集合中的所有文档块（用于统计和调试）"""
        return self._collection.get(
            include=["documents", "metadatas"],
        )

    # ==================== 管理操作 ====================

    def get_stats(self) -> dict:
        """获取集合统计信息"""
        count = self._collection.count()
        return {
            "collection_name": self._collection_name,
            "total_chunks": count,
            "persist_directory": self._persist_dir,
        }

    def delete_collection(self):
        """删除当前集合（用于重建向量库）"""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "AI研发效能智能体知识库"},
        )
        logger.info(f"集合已重建: {self._collection_name}")

    def count(self) -> int:
        """返回当前集合中的文档块数量"""
        return self._collection.count()
