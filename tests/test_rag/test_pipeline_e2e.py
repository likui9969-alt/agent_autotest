"""
RAG 管线端到端测试
==================
验证增量索引、查询、重建等核心流程在管线层面的协同行为。
使用 mock 替换向量库与嵌入模型，不依赖真实 Chroma 服务。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from backend.rag.pipeline import RAGPipeline, _compute_file_hash
from backend.models.rag import RAGQueryRequest


@pytest.fixture
def pipeline(mock_llm_client):
    """返回已替换底层组件的 RAGPipeline"""
    p = RAGPipeline(llm_client=mock_llm_client)

    p.vector_store = MagicMock()
    p.vector_store.add_documents.return_value = ["chunk-id-1"]
    p.vector_store.delete_document.return_value = 0
    p.vector_store.get_file_hashes.return_value = {}

    p.embedder = MagicMock()
    p.embedder.embed_documents.return_value = [[0.1] * 128]

    p.loader = MagicMock()
    p.loader.load.return_value = [
        Document(page_content="测试文档内容", metadata={"filename": "test.txt"})
    ]
    return p


class TestIncrementalIndexing:
    def test_added_and_modified_files(self, pipeline, tmp_path):
        """新增与修改文件应被重新索引，未变文件跳过"""
        old_file = tmp_path / "old.txt"
        old_file.write_text("old content", encoding="utf-8")
        new_file = tmp_path / "new.txt"
        new_file.write_text("new content", encoding="utf-8")

        # 模拟 old.txt 已存在但 hash 不同
        pipeline.vector_store.get_file_hashes.return_value = {
            "old.txt": "different_hash",
        }

        result = pipeline.ingest_directory_incremental(str(tmp_path))

        assert result["added"] == 1
        assert result["modified"] == 1
        assert result["removed"] == 0
        assert result["unchanged"] == 0
        assert result["chunks"] == 2
        pipeline.vector_store.add_documents.assert_called()

    def test_unchanged_files_skipped(self, pipeline, tmp_path):
        """hash 未变文件不应触发重新索引"""
        old_file = tmp_path / "old.txt"
        old_file.write_text("same content", encoding="utf-8")

        real_hash = _compute_file_hash(str(old_file))
        pipeline.vector_store.get_file_hashes.return_value = {
            "old.txt": real_hash,
        }

        result = pipeline.ingest_directory_incremental(str(tmp_path))

        assert result["added"] == 0
        assert result["modified"] == 0
        assert result["removed"] == 0
        assert result["unchanged"] == 1
        assert result["chunks"] == 0
        pipeline.vector_store.add_documents.assert_not_called()

    def test_removed_files_deleted(self, pipeline, tmp_path):
        """已删除文件应从向量库中移除"""
        pipeline.vector_store.get_file_hashes.return_value = {
            "gone.txt": "abc123",
        }

        result = pipeline.ingest_directory_incremental(str(tmp_path))

        assert result["added"] == 0
        assert result["modified"] == 0
        assert result["removed"] == 1
        assert result["unchanged"] == 0
        pipeline.vector_store.delete_document.assert_called_once_with("gone.txt")


class TestPipelineQuery:
    def test_query_returns_structured_response(self, pipeline):
        """完整 RAG 查询应返回结构化响应"""
        pipeline.retriever = MagicMock()
        pipeline.retriever.similarity_search.return_value = [
            Document(
                page_content="登录超时通常由网络延迟引起。",
                metadata={"filename": "login_faq.txt", "score": 0.92},
            )
        ]

        request = RAGQueryRequest(question="登录超时怎么办？", top_k=3)
        response = pipeline.query(request)

        assert response.question == "登录超时怎么办？"
        assert isinstance(response.answer, str)
        assert response.retrieved_count >= 1
        assert response.response_time_ms >= 0


class TestPipelineRebuild:
    def test_rebuild_clears_and_reindexes(self, pipeline, tmp_path):
        """重建应删除集合并重新索引目录"""
        doc_file = tmp_path / "readme.txt"
        doc_file.write_text("rebuild test", encoding="utf-8")

        pipeline.vector_store.delete_collection.return_value = None
        pipeline.loader.load_directory.return_value = [
            Document(page_content="rebuild test", metadata={"filename": "readme.txt"})
        ]

        count = pipeline.rebuild(str(tmp_path))
        assert count >= 1
        pipeline.vector_store.delete_collection.assert_called_once()
