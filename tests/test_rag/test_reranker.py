"""
Reranker 测试
===============
覆盖 LLM 轻量重排序的核心行为：
- 正常重排（LLM 返回合法 JSON）
- 降级路径（LLM 异常 / 非法 JSON / 编号越界 / 有效编号不足）
- 边界（候选不足不调用 LLM / 漏排编号不丢文档）
- pipeline 集成（召回-重排两阶段触发与关闭）
"""
from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from backend.models.rag import RAGQueryRequest
from backend.rag.pipeline import RAGPipeline
from backend.rag.reranker import Reranker


def _make_docs(n: int, prefix: str = "文档") -> list[Document]:
    """构造 n 个候选文档（内容带编号便于断言）"""
    return [
        Document(page_content=f"{prefix}{i}: 相关内容片段", metadata={"filename": f"f{i}.txt"})
        for i in range(n)
    ]


def _llm_returning(text: str) -> MagicMock:
    """构造 chat 返回固定文本的 mock LLM 客户端"""
    client = MagicMock()
    client.chat.return_value = text
    return client


# ==================================================================
# Reranker 单元测试
# ==================================================================

class TestRerankerNormal:
    """正常重排路径"""

    def test_reorder_by_llm_ranking(self):
        """LLM 返回的排序应生效：ranking=[2,0,1] → 文档2 在最前"""
        llm = _llm_returning('{"ranking": [2, 0, 1]}')
        reranker = Reranker(llm_client=llm)
        docs = _make_docs(3)

        ranked = reranker.rerank("查询", docs, top_n=2)

        # top_n=2：只保留重排后前 2 个（文档2、文档0）
        assert len(ranked) == 2
        assert ranked[0].page_content.startswith("文档2")
        assert ranked[1].page_content.startswith("文档0")

    def test_markdown_wrapped_json_parsed(self):
        """LLM 用 markdown 代码块包裹 JSON 时应正确剥离"""
        llm = _llm_returning('```json\n{"ranking": [1, 0]}\n```')
        reranker = Reranker(llm_client=llm)
        ranked = reranker.rerank("查询", _make_docs(2), top_n=1)
        assert ranked[0].page_content.startswith("文档1")

    def test_rerank_metadata_written(self):
        """重排后应写入 rerank_rank / rerank_score 元数据"""
        llm = _llm_returning('{"ranking": [3, 1, 0, 2]}')
        reranker = Reranker(llm_client=llm)
        ranked = reranker.rerank("查询", _make_docs(4), top_n=3)

        assert ranked[0].metadata["rerank_rank"] == 0
        assert ranked[1].metadata["rerank_rank"] == 1
        assert "rerank_score" in ranked[0].metadata

    def test_missing_ids_appended_in_original_order(self):
        """LLM 漏排的候选按原顺序追加（不丢文档）"""
        # 4 个候选只排了 [2, 0]（漏 1、3），有效编号 2 个达标
        llm = _llm_returning('{"ranking": [2, 0]}')
        reranker = Reranker(llm_client=llm)
        ranked = reranker.rerank("查询", _make_docs(4), top_n=3)

        # 漏掉的文档1、文档3 按原顺序追加在已排编号之后
        assert [d.page_content[:3] for d in ranked] == ["文档2", "文档0", "文档1"]

    def test_temperature_is_zero(self):
        """排序任务应使用 temperature=0（确定性输出）"""
        llm = _llm_returning('{"ranking": [0, 1]}')
        reranker = Reranker(llm_client=llm)
        reranker.rerank("查询", _make_docs(2), top_n=1)
        _, kwargs = llm.chat.call_args
        assert kwargs.get("temperature") == 0


class TestRerankerFallback:
    """降级路径：任何 LLM 异常都不应破坏检索"""

    def test_llm_exception_falls_back(self):
        """LLM 调用抛异常 → 返回原始顺序截断"""
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("LLM 不可用")
        reranker = Reranker(llm_client=llm)

        ranked = reranker.rerank("查询", _make_docs(3), top_n=2)

        assert len(ranked) == 2
        assert ranked[0].page_content.startswith("文档0")  # 原始顺序
        assert ranked[1].page_content.startswith("文档1")

    def test_invalid_json_falls_back(self):
        """LLM 返回自然语言（非法 JSON）→ 降级原始顺序"""
        llm = _llm_returning("我认为文档 2 最相关，其次是文档 0。")
        reranker = Reranker(llm_client=llm)

        ranked = reranker.rerank("查询", _make_docs(3), top_n=2)
        assert ranked[0].page_content.startswith("文档0")

    def test_empty_response_falls_back(self):
        """LLM 返回空字符串 → 降级"""
        llm = _llm_returning("")
        reranker = Reranker(llm_client=llm)
        ranked = reranker.rerank("查询", _make_docs(3), top_n=2)
        assert ranked[0].page_content.startswith("文档0")

    def test_out_of_range_ids_ignored(self):
        """越界编号应被忽略（只保留合法编号）"""
        # 3 个候选，编号 5/9 越界，有效仅 [1] < 2 → 不可信降级
        llm = _llm_returning('{"ranking": [5, 9, 1]}')
        reranker = Reranker(llm_client=llm)
        ranked = reranker.rerank("查询", _make_docs(3), top_n=2)
        assert ranked[0].page_content.startswith("文档0")

    def test_too_few_valid_ids_falls_back(self):
        """有效编号不足半数时整体降级（防 LLM 瞎排）"""
        # 4 个候选需 >= 2 个有效编号；ranking 只含 1 个合法
        llm = _llm_returning('{"ranking": [3, 99]}')
        reranker = Reranker(llm_client=llm)
        ranked = reranker.rerank("查询", _make_docs(4), top_n=2)
        assert ranked[0].page_content.startswith("文档0")


class TestRerankerEdgeCases:
    """边界条件"""

    def test_no_llm_call_when_few_candidates(self):
        """候选数 <= top_n 时不应调用 LLM（召回顺序已够用）"""
        llm = _llm_returning('{"ranking": [0]}')
        reranker = Reranker(llm_client=llm)

        ranked = reranker.rerank("查询", _make_docs(3), top_n=5)

        assert len(ranked) == 3
        llm.chat.assert_not_called()

    def test_empty_documents(self):
        """空候选列表直接返回空"""
        reranker = Reranker(llm_client=_llm_returning("{}"))
        assert reranker.rerank("查询", [], top_n=3) == []

    def test_duplicate_ids_counted_once(self):
        """重复编号只取一次，漏掉的追加"""
        llm = _llm_returning('{"ranking": [1, 1, 1, 0]}')
        reranker = Reranker(llm_client=llm)
        ranked = reranker.rerank("查询", _make_docs(4), top_n=3)
        assert [d.page_content[:3] for d in ranked] == ["文档1", "文档0", "文档2"]


# ==================================================================
# Pipeline 集成测试（召回-重排两阶段）
# ==================================================================

class TestPipelineRerankIntegration:
    """验证 RAGPipeline.query 的两阶段检索行为"""

    def _make_pipeline(self, mock_llm_client) -> RAGPipeline:
        """构造绕过 __init__ 的 pipeline（组件全部手动注入 mock）"""
        p = RAGPipeline.__new__(RAGPipeline)
        p.llm_client = mock_llm_client
        p.retriever = MagicMock()
        p.reranker = MagicMock()
        return p

    def test_rerank_triggered_with_more_candidates(self, mock_llm_client):
        """召回数 > top_k 时应触发重排，响应标记 rerank_used=True"""
        p = self._make_pipeline(mock_llm_client)
        candidates = _make_docs(10)
        p.retriever.similarity_search.return_value = candidates
        # mock 重排返回前 3 个（倒序，便于断言）
        p.reranker.rerank.return_value = list(reversed(candidates[:3]))

        request = RAGQueryRequest(question="测试问题", top_k=3)
        response = p.query(request)

        # 召回阶段取 max(top_k=3, RERANK_CANDIDATE_K=10) = 10 个候选
        _, kwargs = p.retriever.similarity_search.call_args
        assert kwargs.get("top_k") == 10
        # 重排被调用，top_n = 用户 top_k
        _, rkwargs = p.reranker.rerank.call_args
        assert rkwargs.get("top_n") == 3
        assert response.rerank_used is True
        assert response.retrieved_count == 3

    def test_rerank_disabled_keeps_single_stage(self, mock_llm_client, monkeypatch):
        """RERANK_ENABLED=False 时退化为单阶段（原行为）"""
        from unittest.mock import patch
        p = self._make_pipeline(mock_llm_client)
        p.retriever.similarity_search.return_value = _make_docs(5)

        request = RAGQueryRequest(question="测试问题", top_k=3)

        # 构造 RERANK_ENABLED=False 的 settings
        fake_settings = MagicMock()
        fake_settings.RERANK_ENABLED = False
        fake_settings.RERANK_CANDIDATE_K = 10
        with patch("backend.rag.pipeline.get_settings", return_value=fake_settings):
            response = p.query(request)

        p.reranker.rerank.assert_not_called()
        assert response.rerank_used is False
        # 召回直接用用户 top_k
        _, kwargs = p.retriever.similarity_search.call_args
        assert kwargs.get("top_k") == 3

    def test_mmr_search_also_reranked(self, mock_llm_client):
        """search_type=mmr 时召回同样走两阶段"""
        p = self._make_pipeline(mock_llm_client)
        p.retriever.mmr_search.return_value = _make_docs(10)
        p.reranker.rerank.return_value = _make_docs(3)

        request = RAGQueryRequest(question="测试问题", top_k=3, search_type="mmr")
        response = p.query(request)

        p.retriever.mmr_search.assert_called_once()
        p.reranker.rerank.assert_called_once()
        assert response.rerank_used is True

    def test_few_results_skip_rerank(self, mock_llm_client):
        """召回结果不足 top_k 时不触发重排（无可排内容）"""
        p = self._make_pipeline(mock_llm_client)
        p.retriever.similarity_search.return_value = _make_docs(2)

        request = RAGQueryRequest(question="测试问题", top_k=5)
        response = p.query(request)

        p.reranker.rerank.assert_not_called()
        assert response.rerank_used is False
        assert response.retrieved_count == 2
