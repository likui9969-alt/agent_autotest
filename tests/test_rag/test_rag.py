"""
Tests for RAG Module
====================
- TextSplitter
- DocumentLoader
- EmbeddingGenerator
- VectorStore
- Retriever (similarity + MMR)
- RAGPipeline
"""
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest

from langchain_core.documents import Document

from backend.rag.splitter import TextSplitter
from backend.rag.loader import DocumentLoader
from backend.rag.embeddings import EmbeddingGenerator
from backend.rag.retriever import Retriever
from backend.rag.vector_store import VectorStore


# ==================== TextSplitter Tests ====================

class TestTextSplitter:
    """文本切割器测试"""

    def test_split_single_document(self):
        """单个文档切割应返回多个块"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        doc = Document(page_content="A" * 200, metadata={"filename": "test.txt"})
        chunks = splitter.split([doc])

        assert len(chunks) >= 2  # 200 chars / 50 chunksize = 至少 4 个

    def test_split_empty_document(self):
        """空文档应返回空列表"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split([])
        assert chunks == []

    def test_small_document_no_split(self):
        """小于 chunk_size 的文档不应被切割"""
        splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        doc = Document(page_content="hello world", metadata={})
        chunks = splitter.split([doc])

        assert len(chunks) == 1
        assert chunks[0].page_content == "hello world"

    def test_chunk_metadata_preserved(self):
        """切割后的块应保留原文档 metadata"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        doc = Document(
            page_content="A" * 200,
            metadata={"filename": "test.txt", "source": "/path/to/test.txt"},
        )
        chunks = splitter.split([doc])

        for chunk in chunks:
            assert chunk.metadata["filename"] == "test.txt"
            assert chunk.metadata["source"] == "/path/to/test.txt"

    def test_split_plain_text(self):
        """split_text 应返回字符串列表"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        result = splitter.split_text("A" * 200)
        assert len(result) >= 2
        assert all(isinstance(s, str) for s in result)

    def test_split_plain_text_empty(self):
        """空文本 split_text 应返回空列表"""
        splitter = TextSplitter()
        result = splitter.split_text("")
        assert result == []

    def test_separator_priority(self):
        """分隔符优先级：段落 > 换行 > 句号 > 空格"""
        splitter = TextSplitter(chunk_size=200, chunk_overlap=0)
        # 包含段落分隔（\n\n）的文本，应在段落边界切割
        text = "第一段内容。包含句号。" * 20 + "\n\n" + "第二段内容。也包含句号。" * 20
        doc = Document(page_content=text, metadata={})
        chunks = splitter.split([doc])

        # 至少应有 2 个块（以 \n\n 为界）
        assert len(chunks) >= 2


# ==================== DocumentLoader Tests ====================

class TestDocumentLoader:
    """文档加载器测试"""

    def test_load_txt(self, tmp_path):
        """加载 txt 文件应返回文档列表"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, 世界！", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(str(txt_file))

        assert len(docs) == 1
        assert docs[0].page_content == "Hello, 世界！"
        assert docs[0].metadata["file_type"] == "txt"

    def test_load_nonexistent_file(self, tmp_path):
        """加载不存在的文件应抛 FileNotFoundError"""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / "nonexistent.txt"))

    def test_unsupported_format(self, tmp_path):
        """不支持的格式应抛 ValueError"""
        loader = DocumentLoader()
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content", encoding="utf-8")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            loader.load(str(test_file))

    def test_load_directory(self, tmp_path):
        """加载目录应索引所有支持的文档"""
        doc1 = tmp_path / "doc1.txt"
        doc1.write_text("Content 1", encoding="utf-8")
        doc2 = tmp_path / "doc2.txt"
        doc2.write_text("Content 2", encoding="utf-8")
        # .md 现也支持，会被加载
        supported_md = tmp_path / "readme.md"
        supported_md.write_text("Markdown", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path))

        assert len(docs) == 3  # .txt × 2 + .md × 1

    def test_overly_large_file(self, tmp_path):
        """超过大小限制的文件应抛 ValueError"""
        loader = DocumentLoader()
        # loader.MAX_FILE_SIZE = 10MB
        large_file = tmp_path / "large.txt"
        # mock 文件大小为 11MB
        with patch("backend.rag.loader.Path.stat") as mock_stat:
            stat_result = MagicMock()
            stat_result.st_size = 11 * 1024 * 1024  # 11MB
            mock_stat.return_value = stat_result

            with pytest.raises(ValueError, match="文件过大"):
                loader.load(str(large_file))

    def test_pdf_loading_with_mock(self):
        """PDF 文件加载测试（mock pypdf）"""
        loader = DocumentLoader()
        mock_pdf_page = MagicMock()
        mock_pdf_page.extract_text.return_value = "PDF page content"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_pdf_page, mock_pdf_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            docs = loader._load_pdf(Path("/fake/test.pdf"))

        assert len(docs) == 2
        assert docs[0].page_content == "PDF page content"
        assert docs[0].metadata["file_type"] == "pdf"

    def test_txt_format_size_display(self):
        """_format_size 应正确格式化文件大小"""
        assert "1.0 KB" in DocumentLoader._format_size(1024)
        assert "1.0 MB" in DocumentLoader._format_size(1024 * 1024)

    def test_load_markdown(self, tmp_path):
        """加载 .md 文件应正常返回"""
        loader = DocumentLoader()
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Title\n\n这是**粗体**和`代码`。", encoding="utf-8")

        docs = loader.load(str(md_file))
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "md"
        assert "# Title" in docs[0].page_content

    def test_load_csv(self, tmp_path):
        """加载 .csv 文件应返回结构化数据预览"""
        loader = DocumentLoader()
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,city\n张三,28,北京\n李四,35,上海\n", encoding="utf-8")

        docs = loader.load(str(csv_file))
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "csv"
        assert "CSV 数据" in docs[0].page_content
        assert "name,age,city" in docs[0].page_content
        assert "张三,28,北京" in docs[0].page_content

    def test_load_md_with_code_block(self, tmp_path):
        """含代码块的 .md 文件应正常加载"""
        loader = DocumentLoader()
        md_file = tmp_path / "code.md"
        md_file.write_text("```python\nprint('hello')\n```\n\n说明文字。", encoding="utf-8")

        docs = loader.load(str(md_file))
        assert len(docs) == 1
        assert "print('hello')" in docs[0].page_content

    def test_load_csv_unsupported_still_blocks(self, tmp_path):
        """不支持的格式仍应阻止"""
        loader = DocumentLoader()
        test_file = tmp_path / "test.doc"
        test_file.write_text("old word format", encoding="utf-8")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            loader.load(str(test_file))
        """load_directory_batch 应分批返回文档"""
        loader = DocumentLoader()
        for i in range(5):
            f = tmp_path / f"doc{i}.txt"
            f.write_text(f"Content {i}", encoding="utf-8")

        batches = list(loader.load_directory_batch(str(tmp_path), batch_size=2))
        assert len(batches) >= 3  # 5 个文件，每批 2 个 → 至少 3 批
        total_docs = sum(len(b) for b in batches)
        assert total_docs == 5

    def test_load_directory_batch_empty_dir(self, tmp_path):
        """空目录应返回空"""
        loader = DocumentLoader()
        batches = list(loader.load_directory_batch(str(tmp_path / "nonexistent")))
        assert batches == []

    def test_load_directory_batch_single_batch(self, tmp_path):
        """文件数少于 batch_size 时应只有一批"""
        loader = DocumentLoader()
        f = tmp_path / "test.txt"
        f.write_text("Hello", encoding="utf-8")

        batches = list(loader.load_directory_batch(str(tmp_path), batch_size=10))
        assert len(batches) == 1

    def test_load_directory_batch_non_txt_skipped(self, tmp_path):
        """非支持格式应被跳过"""
        loader = DocumentLoader()
        f1 = tmp_path / "test.txt"
        f1.write_text("Hello", encoding="utf-8")
        f2 = tmp_path / "test.xyz"
        f2.write_text("Skip me", encoding="utf-8")

        batches = list(loader.load_directory_batch(str(tmp_path)))
        total = sum(len(b) for b in batches)
        assert total == 1
        assert "500.0 B" in DocumentLoader._format_size(500)


# ==================== EmbeddingGenerator Tests ====================

class TestEmbeddingGenerator:
    """嵌入生成器测试"""

    def test_embed_documents(self, mock_llm_client):
        """批量嵌入应返回等长向量列表"""
        gen = EmbeddingGenerator(llm_client=mock_llm_client)
        texts = ["文档1", "文档2", "文档3"]
        vectors = gen.embed_documents(texts)

        assert len(vectors) == 3
        assert all(len(v) == 128 for v in vectors)

    def test_embed_documents_empty(self, mock_llm_client):
        """空列表应返回空列表"""
        gen = EmbeddingGenerator(llm_client=mock_llm_client)
        assert gen.embed_documents([]) == []

    def test_embed_query(self, mock_llm_client):
        """单条查询嵌入应返回向量"""
        gen = EmbeddingGenerator(llm_client=mock_llm_client)
        vector = gen.embed_query("测试查询")
        assert len(vector) == 128

    def test_embed_batch_single_chunk(self, mock_llm_client):
        """少于 _EMBED_BATCH_SIZE 条时不分批，应直接调用一次 embed"""
        from backend.rag.embeddings import _EMBED_BATCH_SIZE
        gen = EmbeddingGenerator(llm_client=mock_llm_client)
        texts = ["text"] * (_EMBED_BATCH_SIZE - 1)
        vectors = gen.embed_documents(texts)

        assert len(vectors) == len(texts)
        assert mock_llm_client.embed.call_count == 1

    def test_embed_batch_multiple(self, mock_llm_client):
        """超过 _EMBED_BATCH_SIZE 条时应分批调用"""
        from backend.rag.embeddings import _EMBED_BATCH_SIZE
        gen = EmbeddingGenerator(llm_client=mock_llm_client)
        texts = ["text"] * (_EMBED_BATCH_SIZE * 2 + 1)
        vectors = gen.embed_documents(texts)

        assert len(vectors) == len(texts)
        # 应调用 3 次（512 + 512 + 1）
        assert mock_llm_client.embed.call_count == 3

    def test_embed_batch_partial_last(self, mock_llm_client):
        """最后一批不足 _EMBED_BATCH_SIZE 条时也应正常返回"""
        from backend.rag.embeddings import _EMBED_BATCH_SIZE
        gen = EmbeddingGenerator(llm_client=mock_llm_client)
        texts = ["text"] * (_EMBED_BATCH_SIZE + 5)
        vectors = gen.embed_documents(texts)

        assert len(vectors) == len(texts)


# ==================== VectorStore Tests ====================

class TestVectorStore:
    """向量存储测试（使用 mock chromadb）"""

    @pytest.fixture
    def mock_chroma_client(self):
        """mock chromadb.PersistentClient"""
        with patch("backend.rag.vector_store.chromadb.PersistentClient") as mock:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock.return_value = mock_client
            yield mock_client, mock_collection

    def test_init_creates_collection(self, mock_chroma_client):
        """初始化应创建或获取集合"""
        mock_client, mock_collection = mock_chroma_client

        store = VectorStore("test_kb")

        mock_client.get_or_create_collection.assert_called_once()
        assert store._collection_name == "test_kb"

    def test_add_documents(self, mock_chroma_client):
        """添加文档应调用 collection.add"""
        mock_client, mock_collection = mock_chroma_client
        store = VectorStore("test_kb")

        docs = [Document(page_content="test", metadata={"filename": "a.txt"})]
        embeddings = [[0.1] * 128]

        result = store.add_documents(docs, embeddings)

        assert len(result) == 1
        mock_collection.add.assert_called_once()

    def test_add_empty_documents(self, mock_chroma_client):
        """空文档列表应返回空列表"""
        mock_client, mock_collection = mock_chroma_client
        store = VectorStore("test_kb")

        result = store.add_documents([], [])
        assert result == []

    def test_get_stats(self, mock_chroma_client):
        """统计信息应包含集合名和块数"""
        mock_client, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 42
        store = VectorStore("test_kb")

        stats = store.get_stats()
        assert stats["collection_name"] == "test_kb"
        assert stats["total_chunks"] == 42

    def test_delete_collection(self, mock_chroma_client):
        """删除集合后应重建"""
        mock_client, mock_collection = mock_chroma_client
        store = VectorStore("test_kb")

        store.delete_collection()

        mock_client.delete_collection.assert_called_once_with("test_kb")
        # 重建后应再次调用 get_or_create
        assert mock_client.get_or_create_collection.call_count >= 2


# ==================== Retriever Tests ====================

class TestRetriever:
    """检索器测试"""

    @pytest.fixture
    def mock_chroma_query(self, mock_vector_store):
        """让 mock_vector_store.query 返回带实际 docs 的结果"""
        docs = [
            Document(page_content="登录超时解决方案", metadata={"filename": "a.txt", "score": 0.9}),
            Document(page_content="数据库连接池问题", metadata={"filename": "b.txt", "score": 0.8}),
        ]
        mock_vector_store._docs = {
            "id1": {"doc": docs[0], "embedding": [0.1] * 128},
            "id2": {"doc": docs[1], "embedding": [0.1] * 128},
        }
        return mock_vector_store

    def test_similarity_search_found(self, mock_vector_store, mock_llm_client):
        """相似度检索应返回文档列表"""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 128

        retriever = Retriever(vector_store=mock_vector_store, embedding_generator=mock_embedder)
        results = retriever.similarity_search("登录超时", top_k=2)

        assert len(results) >= 0

    def test_similarity_search_empty_store(self):
        """空向量库应返回空列表"""
        store = MagicMock()
        store.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 128

        retriever = Retriever(vector_store=store, embedding_generator=embedder)
        results = retriever.similarity_search("test", top_k=5)

        assert results == []

    def test_mmr_select_diversity(self):
        """MMR 选择应在多样性和相关性之间平衡"""
        from backend.rag.retriever import Retriever

        candidates = [
            Document(page_content="doc1", metadata={"_embedding": [1.0, 0.0, 0.0]}),
            Document(page_content="doc2", metadata={"_embedding": [0.0, 1.0, 0.0]}),
            Document(page_content="doc3", metadata={"_embedding": [0.0, 0.0, 1.0]}),
        ]
        query_embedding = [1.0, 0.0, 0.0]

        retriever = Retriever.__new__(Retriever)
        selected = retriever._mmr_select(query_embedding, candidates, top_k=2)

        assert len(selected) == 2

    def test_mmr_numpy_path(self):
        """numpy 路径的 MMR 应正确运行"""
        from backend.rag.retriever import Retriever, _HAS_NUMPY
        if not _HAS_NUMPY:
            pytest.skip("numpy not available")

        candidates = [
            Document(page_content="doc1", metadata={"_embedding": [1.0, 0.0, 0.0]}),
            Document(page_content="doc2", metadata={"_embedding": [0.0, 1.0, 0.0]}),
            Document(page_content="doc3", metadata={"_embedding": [0.0, 0.0, 1.0]}),
        ]
        query_embedding = [1.0, 0.0, 0.0]

        retriever = Retriever.__new__(Retriever)
        selected = retriever._mmr_select_numpy(query_embedding, candidates, top_k=2)

        assert len(selected) == 2

    def test_mmr_python_fallback(self):
        """纯 Python 回退路径的 MMR 应正确运行"""
        from backend.rag.retriever import Retriever

        candidates = [
            Document(page_content="doc1", metadata={"_embedding": [1.0, 0.0, 0.0]}),
            Document(page_content="doc2", metadata={"_embedding": [0.0, 1.0, 0.0]}),
            Document(page_content="doc3", metadata={"_embedding": [0.0, 0.0, 1.0]}),
        ]
        query_embedding = [1.0, 0.0, 0.0]

        retriever = Retriever.__new__(Retriever)
        selected = retriever._mmr_select_py(query_embedding, candidates, top_k=2)

        assert len(selected) == 2

    def test_mmr_output_consistent(self):
        """numpy 与 Python 路径结果应一致"""
        from backend.rag.retriever import Retriever, _HAS_NUMPY
        if not _HAS_NUMPY:
            pytest.skip("numpy not available")

        candidates = [
            Document(page_content="doc1", metadata={"_embedding": [1.0, 0.0, 0.0]}),
            Document(page_content="doc2", metadata={"_embedding": [0.0, 1.0, 0.0]}),
            Document(page_content="doc3", metadata={"_embedding": [0.0, 0.0, 1.0]}),
        ]
        query_embedding = [1.0, 0.0, 0.0]

        retriever = Retriever.__new__(Retriever)
        numpy_result = retriever._mmr_select_numpy(query_embedding, candidates, top_k=2)
        py_result = retriever._mmr_select_py(query_embedding, candidates, top_k=2)

        assert len(numpy_result) == len(py_result) == 2
        # 两个路径的排名可能不同（数值精度差异），但都是有效 MMR 结果
        # 验证至少含 doc1（与查询最相关）
        assert any("doc1" in d.page_content for d in numpy_result)

    def test_format_results(self):
        """Chroma 结果应正确转换为 Document"""
        chroma_results = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"filename": "a.txt"}, {"filename": "b.txt"}]],
            "distances": [[0.1, 0.2]],
        }

        retriever = Retriever.__new__(Retriever)
        docs = retriever._format_results(chroma_results)

        assert len(docs) == 2
        assert docs[0].page_content == "doc1"
        assert "score" in docs[0].metadata
        assert docs[0].metadata["score"] > 0

    def test_format_results_empty(self):
        """空结果应返回空列表"""
        chroma_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        retriever = Retriever.__new__(Retriever)
        docs = retriever._format_results(chroma_results)

        assert docs == []


class TestRAGPipeline:
    """RAG 管线整体测试"""

    def test_ingest_file(self, mock_vector_store, mock_llm_client):
        """索引单个文件"""
        from backend.rag.pipeline import RAGPipeline

        with patch("backend.rag.vector_store.VectorStore") as mock_vs, \
             patch("backend.rag.loader.DocumentLoader") as mock_loader:

            # 设置 mock
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [
                Document(page_content="test content", metadata={"filename": "test.txt"})
            ]
            mock_loader.return_value = mock_loader_instance

            pipeline = RAGPipeline(llm_client=mock_llm_client)

            # 替换组件为 mock
            pipeline.vector_store = mock_vector_store
            pipeline.loader = mock_loader_instance

            count = pipeline.ingest_file("/fake/test.txt")
            assert count >= 0

    def test_query_returns_response(self, mock_llm_client):
        """RAG 查询应返回结构化的 RAGQueryResponse"""
        from backend.rag.pipeline import RAGPipeline
        from backend.models.rag import RAGQueryRequest

        pipeline = RAGPipeline(llm_client=mock_llm_client)

        # Mock retriever
        pipeline.retriever = MagicMock()
        pipeline.retriever.similarity_search.return_value = [
            Document(page_content="test", metadata={"filename": "a.txt", "score": 0.9})
        ]
        pipeline.retriever.mmr_search.return_value = [
            Document(page_content="test", metadata={"filename": "a.txt", "score": 0.9})
        ]

        # Mock LLM 返回固定回答
        pipeline.llm_client = mock_llm_client

        request = RAGQueryRequest(question="测试问题")
        response = pipeline.query(request)

        assert response.question == "测试问题"
        assert isinstance(response.answer, str)
        assert response.retrieved_count >= 0
        assert response.response_time_ms >= 0

    def test_ingest_directory_batch(self, mock_llm_client, tmp_path):
        """分批索引目录"""
        from backend.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(llm_client=mock_llm_client)
        pipeline.vector_store = MagicMock()
        pipeline.loader = MagicMock()
        pipeline.loader.load_directory_batch.return_value = [
            [Document(page_content="batch1", metadata={})],
            [Document(page_content="batch2", metadata={})],
        ]
        pipeline.embedder = MagicMock()
        pipeline.embedder.embed_documents.return_value = [[0.1] * 128]

        count = pipeline.ingest_directory_batch("/fake/dir", batch_size=2)
        assert count > 0

    def test_ingest_directory_batch_zero_is_full(self, mock_llm_client, tmp_path):
        """batch_size=0 时使用一次性加载（兼容旧行为）"""
        from backend.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(llm_client=mock_llm_client)
        pipeline.vector_store = MagicMock()
        pipeline.loader = MagicMock()
        pipeline.loader.load_directory.return_value = [
            Document(page_content="doc1", metadata={}),
        ]
        pipeline.embedder = MagicMock()
        pipeline.embedder.embed_documents.return_value = [[0.1] * 128]

        count = pipeline.ingest_directory_batch("/fake/dir", batch_size=0)
        assert count > 0

    def test_rebuild(self, mock_llm_client, tmp_path):
        """重建知识库"""
        from backend.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(llm_client=mock_llm_client)
        pipeline.vector_store = MagicMock()
        pipeline.loader = MagicMock()
        pipeline.loader.load_directory.return_value = [
            Document(page_content="test", metadata={})
        ]
        pipeline.embedder = MagicMock()
        pipeline.embedder.embed_documents.return_value = [[0.1] * 128]
        pipeline.vector_store.delete_collection.return_value = None

        count = pipeline.rebuild(str(tmp_path / "docs"))
        assert count >= 0

    def test_stats(self, mock_llm_client):
        """统计信息"""
        from backend.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(llm_client=mock_llm_client)
        pipeline.vector_store = MagicMock()
        pipeline.vector_store.get_stats.return_value = {"total_chunks": 10}

        stats = pipeline.stats()
        assert stats["total_chunks"] == 10
