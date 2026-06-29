"""
RAG 管线编排模块
将文档加载→切割→嵌入→存储→检索→生成的完整流程串联起来
"""
import logging
import time
from pathlib import Path

from backend.config.settings import get_settings
from backend.rag.loader import DocumentLoader
from backend.rag.splitter import TextSplitter
from backend.rag.embeddings import EmbeddingGenerator
from backend.rag.vector_store import VectorStore
from backend.rag.retriever import Retriever
from backend.llm.client import LLMClient
from backend.llm.prompts import get_template
from backend.models.rag import RAGQueryRequest, RAGQueryResponse, SourceCitation

from backend.api.deps import get_llm_client

logger = logging.getLogger("ai_rd_agent")


class RAGPipeline:
    """RAG 完整管线

    管线流程：
    1. 文档加载   — 从目录/文件读取原始文档
    2. 文本切割   — 将长文档切分为语义块
    3. 向量嵌入   — 将文本块转为嵌入向量
    4. 向量存储   — 将向量写入 Chroma
    5. 检索       — 根据用户查询检索相关块
    6. 生成       — 将检索结果传给 LLM 生成回答

    使用示例：
        pipeline = RAGPipeline()
        pipeline.ingest_directory("data/docs/")        # 索引文档
        response = pipeline.query("登录接口500错误怎么办")  # RAG 问答
    """

    def __init__(self, llm_client: LLMClient | None = None):
        """初始化管线各组件（共享 LLMClient 实例以减少资源占用）"""
        # 优先使用注入的 LLM 客户端，否则从全局单例获取
        self.llm_client = llm_client or get_llm_client()
        # 各组件初始化，EmbeddingGenerator 使用共享的 LLMClient
        self.loader = DocumentLoader()
        self.splitter = TextSplitter()
        self.embedder = EmbeddingGenerator(llm_client=self.llm_client)
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store, self.embedder)

        logger.info("RAG 管线已初始化")

    # ==================== 文档索引 ====================

    def ingest_file(self, file_path: str) -> int:
        """索引单个文档文件

        Args:
            file_path: 文档文件路径

        Returns:
            生成的块数量
        """
        logger.info(f"开始索引文档: {file_path}")

        # 1. 加载文档
        documents = self.loader.load(file_path)
        if not documents:
            logger.warning("文档加载后为空")
            return 0

        # 2. 切割文档
        chunks = self.splitter.split(documents)
        if not chunks:
            logger.warning("文档切割后为空")
            return 0

        # 3. 生成嵌入向量
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)

        # 4. 写入向量库
        self.vector_store.add_documents(chunks, embeddings)

        logger.info(f"文档索引完成: {file_path} → {len(chunks)} 个块")
        return len(chunks)

    def ingest_directory(self, dir_path: str) -> int:
        """索引目录下所有支持的文档（一次性加载）

        Args:
            dir_path: 文档目录路径

        Returns:
            总共生成的块数量
        """
        return self.ingest_directory_batch(dir_path, batch_size=0)

    def ingest_directory_batch(self, dir_path: str, batch_size: int = 10) -> int:
        """分批索引目录下所有支持的文档（流式处理）

        每批次加载 batch_size 个文件，执行切割→嵌入→存储后立即释放内存，
        避免大目录下全量加载到内存。

        Args:
            dir_path: 文档目录路径
            batch_size: 每批文件数。设为 0 或 None 时一次性加载所有文件（兼容旧行为）

        Returns:
            总共生成的块数量
        """
        logger.info(f"开始批量索引目录: {dir_path}")

        if not batch_size:
            # 一次性加载（兼容旧行为）
            documents = self.loader.load_directory(dir_path)
            if not documents:
                logger.warning("目录中没有找到支持的文档")
                return 0

            chunks = self.splitter.split(documents)
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embedder.embed_documents(texts)
            self.vector_store.add_documents(chunks, embeddings)
            logger.info(f"批量索引完成: {len(documents)} 个文件 → {len(chunks)} 个块")
            return len(chunks)

        # 分批流式处理
        total_chunks = 0
        batch_count = 0

        for batch_docs in self.loader.load_directory_batch(dir_path, batch_size=batch_size):
            if not batch_docs:
                continue

            batch_count += 1
            chunks = self.splitter.split(batch_docs)

            if chunks:
                texts = [chunk.page_content for chunk in chunks]
                try:
                    embeddings = self.embedder.embed_documents(texts)
                    self.vector_store.add_documents(chunks, embeddings)
                    total_chunks += len(chunks)
                    logger.info(
                        f"  批次 {batch_count} 完成: {len(batch_docs)} 文件 → {len(chunks)} 块 "
                        f"(累计: {total_chunks} 块)"
                    )
                except Exception as e:
                    logger.error(f"批次 {batch_count} 处理失败: {e}")
                    # 继续处理下一批

        logger.info(f"批量索引完成: {batch_count} 批次, 共 {total_chunks} 个块")
        return total_chunks

    # ==================== RAG 查询 ====================

    def query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """执行 RAG 查询：检索 + 生成

        Args:
            request: 查询请求（问题 + 检索参数）

        Returns:
            RAG 查询响应（回答 + 来源引用）
        """
        start_time = time.time()

        # 1. 检索相关文档块
        if request.search_type == "mmr":
            retrieved = self.retriever.mmr_search(
                query=request.question,
                top_k=request.top_k,
            )
        else:
            retrieved = self.retriever.similarity_search(
                query=request.question,
                top_k=request.top_k,
            )

        # 2. 构建上下文
        context_parts = []
        sources = []
        for i, doc in enumerate(retrieved):
            context_parts.append(f"[文档{i+1}] {doc.page_content}")
            if request.include_sources:
                excerpt = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                sources.append(SourceCitation(
                    source_file=doc.metadata.get("filename", "未知"),
                    chunk_index=doc.metadata.get("chunk_index", i),
                    excerpt=excerpt,
                    score=doc.metadata.get("score", 0.0),
                ))

        context = "\n\n".join(context_parts) if context_parts else "未找到相关知识库内容。"

        # 3. 构建 Prompt
        template = get_template("rag_query")
        messages = [
            {"role": "system", "content": template.system},
            {"role": "user", "content": template.user.format(
                question=request.question,
                context=context,
            )},
        ]

        # 4. 调用 LLM 生成回答
        answer = self.llm_client.chat(
            messages=messages,
            temperature=template.temperature,
        )

        # 5. 计算耗时
        elapsed_ms = (time.time() - start_time) * 1000

        # 6. 构建响应
        return RAGQueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            retrieved_count=len(retrieved),
            response_time_ms=round(elapsed_ms, 1),
        )

    # ==================== 知识库管理 ====================

    def rebuild(self, dir_path: str | None = None) -> int:
        """重建向量库：删除旧数据，重新索引

        Args:
            dir_path: 文档目录路径（不传则从配置读取）

        Returns:
            重新索引的块数量
        """
        settings = get_settings()
        target_dir = dir_path or settings.get_upload_dir()

        logger.info("正在重建向量库...")
        self.vector_store.delete_collection()
        # 重建后需要重建 retriever 中的引用
        self.retriever = Retriever(self.vector_store, self.embedder)

        if Path(target_dir).exists():
            return self.ingest_directory(target_dir)
        return 0

    def delete_document(self, filename: str) -> int:
        """按文件名删除知识库中的文档

        Args:
            filename: 文档文件名

        Returns:
            删除的块数量
        """
        return self.vector_store.delete_document(filename)

    def get_documents(self) -> list[dict]:
        """列出知识库中的所有文档

        Returns:
            [{filename, chunk_count}, ...]
        """
        return self.vector_store.get_documents()

    def stats(self) -> dict:
        """获取知识库统计信息"""
        return self.vector_store.get_stats()
