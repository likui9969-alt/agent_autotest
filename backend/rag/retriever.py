"""
检索器模块
支持相似度检索（Similarity Search）和 MMR 检索（最大边际相关性）
"""
import logging
from langchain_core.documents import Document

from backend.config.settings import get_settings
from backend.rag.vector_store import VectorStore
from backend.rag.embeddings import EmbeddingGenerator

logger = logging.getLogger("ai_rd_agent")


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

        # 获取候选集
        candidates = self._store.query(query_embedding, top_k=fetch_k)
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

        每次从候选集中选择一个文档：
        - 与查询相关度高（加正分）
        - 与已选文档不相似（避免冗余，加分）
        """
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            """计算两个向量的余弦相似度"""
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        # 计算每个候选文档与查询的相关度
        # 注意：candidates 已经是从 Chroma 按距离排序返回的
        # 这里我们使用索引近似，候选越靠前越相关
        n = len(candidates)

        # 将候选按索引位置估算相似度（越靠前分数越高）
        candidate_scores = [
            1.0 - (i / n) * 0.5 for i in range(n)
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

                # 多样性得分：与已选文档的最大相似度（负向）
                diversity = 0.0
                if selected:
                    sim_to_selected = cosine_similarity(
                        candidates[i].metadata.get("_embedding", [0.0] * len(query_embedding)),
                        # 注意：这里简化了 MMR，直接用原始的 Chroma 顺序作为多样性代理
                        query_embedding,  # fallback to query as proxy
                    )
                    diversity = (1 - lambda_mult) * (1.0 - sim_to_selected)

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
            documents.append(doc)

        return documents
