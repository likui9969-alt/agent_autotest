"""
Reranker 模块 — LLM 轻量重排序
召回-重排两阶段检索的第二阶段：
    向量召回 top_k 候选 → LLM 按查询相关性重排 → 取前 top_n

设计权衡（vs Cross-Encoder 重排）：
- 无需额外部署重排模型（cross-encoder 需要独立 GPU 推理服务）
- 复用现有 LLM Provider 链路（自动获得重试 / 熔断 / 多 Provider 回退）
- 代价：增加一次 LLM 调用延迟（约 1~3s），适合对延迟不敏感的知识库问答场景

可靠性设计：
- LLM 返回非法 JSON / 编号越界 / 调用异常时，优雅降级为召回原始顺序
- LLM 漏排的候选按原顺序追加在末尾（不丢文档）
"""
import json
import logging
import re
import time

from langchain_core.documents import Document

from backend.llm.client import LLMClient

logger = logging.getLogger("ai_rd_agent")

# 每个候选注入 prompt 的摘要长度（控制 token）
_SNIPPET_LENGTH = 200

_RERANK_SYSTEM_PROMPT = """你是搜索相关性评估专家。给定用户查询和一组编号的候选文档，
按与查询的相关性从高到低排序。

规则：
1. 只输出纯 JSON，不要任何解释、markdown 代码块或其他文本
2. 输出格式：{"ranking": [编号, 编号, ...]}，编号是候选文档的序号（从 0 开始）
3. ranking 必须包含全部候选编号，最相关的在最前
4. 相关性判断标准：文档是否直接回答查询的问题，而非仅共享主题词"""

_RERANK_USER_PROMPT = """用户查询：{query}

候选文档：
{candidates}

按相关性从高到低输出全部编号的 JSON 排序："""


class Reranker:
    """LLM 轻量重排序器

    使用示例：
        reranker = Reranker(llm_client)
        ranked = reranker.rerank("登录超时怎么办", candidates, top_n=3)
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """按查询相关性重排候选文档

        Args:
            query: 用户查询文本
            documents: 召回阶段获得的候选文档（按向量相似度排序）
            top_n: 重排后保留的文档数（默认全部保留，仅调整顺序）

        Returns:
            重排后的 Document 列表（截断至 top_n）。
            LLM 调用失败或返回非法结果时，返回原始顺序截断至 top_n。
        """
        n = len(documents)
        keep = top_n if top_n is not None else n
        if n == 0:
            return []
        # 候选数不超过保留数时无需重排（顺序已由召回阶段确定）
        if n <= keep:
            return list(documents)

        start = time.time()
        ranking = self._get_ranking_from_llm(query, documents)

        if ranking is None:
            # 降级：原始顺序
            logger.warning("Rerank 降级：使用召回原始顺序")
            return list(documents[:keep])

        # 按编号重排；LLM 漏掉的候选按原顺序追加（防丢失）
        seen = set(ranking)
        ordered_indices = ranking + [i for i in range(n) if i not in seen]
        ranked = [documents[i] for i in ordered_indices]

        # 写入重排元数据（演示与调试可观测）
        for rank, doc in enumerate(ranked):
            doc.metadata["rerank_rank"] = rank
            doc.metadata["rerank_score"] = round(1.0 - rank / max(n, 1), 4)

        elapsed = (time.time() - start) * 1000
        logger.info(
            f"Rerank 完成: {n} 个候选 → 保留 {min(keep, len(ranked))} 个, "
            f"排序头部变化={ranking[0] != 0}, 耗时 {elapsed:.0f}ms"
        )
        return ranked[:keep]

    # ==================== 内部方法 ====================

    def _get_ranking_from_llm(
        self, query: str, documents: list[Document]
    ) -> list[int] | None:
        """调用 LLM 获取相关性排序

        Returns:
            编号列表（合法且覆盖候选集），失败返回 None
        """
        candidates_text = "\n".join(
            f"[{i}] {doc.page_content[:_SNIPPET_LENGTH].strip()}"
            for i, doc in enumerate(documents)
        )
        messages = [
            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": _RERANK_USER_PROMPT.format(
                query=query, candidates=candidates_text,
            )},
        ]

        try:
            # temperature=0：排序任务要求确定性输出
            response = self.llm_client.chat(messages=messages, temperature=0)
        except Exception as e:
            logger.warning(f"Rerank LLM 调用失败: {e}")
            return None

        return self._parse_ranking(response, len(documents))

    def _parse_ranking(self, response: str, n_candidates: int) -> list[int] | None:
        """解析 LLM 返回的排序 JSON，校验编号合法性

        Returns:
            合法编号列表；解析失败或编号非法返回 None
        """
        if not response:
            return None

        # 剥离可能存在的 markdown 代码块包裹（LLM 常见行为）
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Rerank 返回非法 JSON: {response[:200]}")
            return None

        raw = data.get("ranking") if isinstance(data, dict) else None
        if not isinstance(raw, list) or not raw:
            logger.warning(f"Rerank 返回缺少 ranking 字段: {str(data)[:200]}")
            return None

        # 校验：编号为整数、在候选范围内、无重复
        valid: list[int] = []
        seen: set[int] = set()
        for item in raw:
            if not isinstance(item, int) or isinstance(item, bool):
                continue
            if 0 <= item < n_candidates and item not in seen:
                valid.append(item)
                seen.add(item)

        # 有效编号不足一半时视为不可信，降级
        if len(valid) < (n_candidates + 1) // 2:
            logger.warning(
                f"Rerank 有效编号 {len(valid)}/{n_candidates} 不足半数，降级"
            )
            return None
        return valid
