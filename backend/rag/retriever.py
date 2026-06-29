"""
检索器模块
支持相似度检索（Similarity Search）和 MMR 检索（最大边际相关性）
MMR 优先使用 numpy 加速余弦相似度计算，不可用时回退到纯 Python 实现
"""
import logging
import math
from typing import Callable
from langchain_core.documents import Document

from backend.config.settings import get_settings
from backend.rag.vector_store import VectorStore
from backend.rag.embeddings import EmbeddingGenerator

logger = logging.getLogger("ai_rd_agent")

# 尝试导入 numpy（MMR 加速），不可用时用纯 Python fallback
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False
    logger.info("numpy 未安装，MMR 检索将使用纯 Python 实现")


def _cosine_similarity_py(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度计算"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Retriever:
    """RAG 检索器

    提供两种检索策略：
    1. similarity_search() — 相似度检索：返回与查询最相似的文档块
    2. mmr_search() — MMR 检索：在相似度和多样性之间取得平衡

    使用示例：
        retriever = Retriever()
        results = retriever.similarity_search("登录接口500错误", top_k=5)
        results = retriever.mmr_search("登录接口500错误", top_k=5)
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """
        Args:
            vector_store: Chroma 向量存储实例
            embedding_generator: 嵌入生成器实例
        """
        settings = get_settings()
        self._store = vector_store or VectorStore()
        self._embedder = embedding_generator or EmbeddingGenerator()
        self._default_top_k = settings.RETRIEVER_TOP_K

    # ==================== 相似度检索 ====================

    def similarity_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """相似度检索 — 查找与查询向量最相似的文档块

        原理：计算查询向量与所有文档向量的余弦相似度，返回最相似的 top_k 个。

        Args:
            query: 查询文本
            top_k: 返回结果数（默认从配置读取）

        Returns:
            最相关的 LangChain Document 列表
        """
        k = top_k or self._default_top_k

        # 将查询文本转为向量
        query_embedding = self._embedder.embed_query(query)

        # 在 Chroma 中检索
        results = self._store.query(query_embedding, top_k=k)

        # 转换为 LangChain Document 列表
        return self._format_results(results)

    # ==================== MMR 检索 ====================

    def mmr_search(
        self,
        query: str,
        top_k: int | None = None,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """MMR（最大边际相关性）检索

        MMR 在相关性和多样性之间取得平衡：
        - lambda_mult=1.0：等同于纯相似度检索
        - lambda_mult=0.0：最大化多样性
        - lambda_mult=0.5（默认）：平衡相关性和多样性

        Args:
            query: 查询文本
            top_k: 最终返回结果数
            lambda_mult: 多样性控制参数（0~1）

        Returns:
            MMR 选取的 Document 列表
        """
        k = top_k or self._default_top_k

        # MMR 需要先获取更多候选，再从中挑选
        fetch_k = k * 4  # 先取 4 倍的候选

        query_embedding = self._embedder.embed_query(query)

        # 获取候选集（含嵌入向量，用于 MMR 去重）
        candidates = self._store.query(query_embedding, top_k=fetch_k, include_embeddings=True)
        candidate_docs = self._format_results(candidates)

        if len(candidate_docs) <= k:
            return candidate_docs

        # 实现 MMR 选择算法
        selected = self._mmr_select(
            query_embedding=query_embedding,
            candidates=candidate_docs,
            top_k=k,
            lambda_mult=lambda_mult,
        )

        return selected

    def _mmr_select(
        self,
        query_embedding: list[float],
        candidates: list[Document],
        top_k: int,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """MMR 贪心选择算法

        numpy 可用时使用矩阵运算加速，不可用时回退到纯 Python 实现。
        """
        if _HAS_NUMPY:
            return self._mmr_select_numpy(query_embedding, candidates, top_k, lambda_mult)
        return self._mmr_select_py(query_embedding, candidates, top_k, lambda_mult)

    def _mmr_select_numpy(
        self,
        query_embedding: list[float],
        candidates: list[Document],
        top_k: int,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """MMR — numpy 向量化实现"""
        n = len(candidates)
        k = min(top_k, n)

        # 提取所有候选的嵌入向量 → (n, d) 矩阵
        emb_list: list[list[float]] = [
            c.metadata.get("_embedding", query_embedding)
            for c in candidates
        ]
        emb_matrix = np.array(emb_list, dtype=np.float64)
        q_vec = np.array(query_embedding, dtype=np.float64)

        # 计算余弦相似度矩阵
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_normed = emb_matrix / norms

        q_norm = np.linalg.norm(q_vec)
        q_normed = q_vec / q_norm if q_norm > 0 else q_vec

        # 候选-查询相似度 (n,) — 相关度
        sim_to_query = emb_normed @ q_normed

        # 候选-候选相似度矩阵 (n, n) — 用于多样性计算
        sim_matrix = emb_normed @ emb_normed.T

        selected_indices: list[int] = []
        remaining = list(range(n))

        for _ in range(k):
            if not remaining:
                break

            best_score = -1.0
            best_idx = -1

            for i in remaining:
                # 相关性项
                relevance = lambda_mult * sim_to_query[i]

                # 多样性项
                diversity = 0.0
                if selected_indices:
                    max_to_selected = max(sim_matrix[i, j] for j in selected_indices)
                    diversity = (1 - lambda_mult) * (1.0 - max_to_selected)

                score = relevance + diversity
                if score > best_score:
                    best_score = score
                    best_idx = i

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def _mmr_select_py(
        self,
        query_embedding: list[float],
        candidates: list[Document],
        top_k: int,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """MMR — 纯 Python 实现（回退路径）"""
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        n = len(candidates)

        # 相关度：候选文档与查询的余弦相似度
        candidate_scores = [
            cosine_similarity(
                candidates[i].metadata.get("_embedding", query_embedding),
                query_embedding,
            ) for i in range(n)
        ]

        selected = []
        selected_indices = set()

        for _ in range(min(top_k, n)):
            best_score = -float("inf")
            best_idx = -1

            for i in range(n):
                if i in selected_indices:
                    continue

                # 相关性得分
                relevance = lambda_mult * candidate_scores[i]

                # 多样性得分：与已选文档的最大相似度（转换为惩罚项）
                diversity = 0.0
                if selected:
                    max_sim_to_selected = max(
                        cosine_similarity(
                            candidates[i].metadata.get("_embedding", query_embedding),
                            candidates[j].metadata.get("_embedding", query_embedding),
                        )
                        for j in selected_indices
                    )
                    diversity = (1 - lambda_mult) * (1.0 - max_sim_to_selected)

                score = relevance + diversity

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                selected.append(candidates[best_idx])
                selected_indices.add(best_idx)

        return selected

    # ==================== 辅助方法 ====================

    def _format_results(self, chroma_results: dict) -> list[Document]:
        """将 Chroma 查询结果转换为 LangChain Document 列表

        Args:
            chroma_results: Chroma 查询返回的原始字典

        Returns:
            Document 列表
        """
        documents = []

        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return documents

        ids_list = chroma_results["ids"][0]
        docs_list = chroma_results.get("documents", [[]])[0]
        meta_list = chroma_results.get("metadatas", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]
        embeddings_list = chroma_results.get("embeddings", [[]])[0] if chroma_results.get("embeddings") else []

        for i, chunk_id in enumerate(ids_list):
            doc = Document(
                page_content=docs_list[i] if i < len(docs_list) else "",
                metadata={
                    **(meta_list[i] if i < len(meta_list) else {}),
                    "chunk_id": chunk_id,
                    # 将距离转换为相似度分数 [0, 1]
                    # 兼容 L2 和 Cosine 两种度量
                    "score": round(1.0 / (1.0 + distances[i]), 4) if i < len(distances) else 0.0,
                }
            )
            # 保存实际嵌入向量（MMR 多样性计算需要）
            if embeddings_list and i < len(embeddings_list):
                doc.metadata["_embedding"] = embeddings_list[i]
            documents.append(doc)

        return documents
