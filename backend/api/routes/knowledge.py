"""
知识库管理路由
POST   /api/v1/knowledge/upload      — 上传文档并索引
GET    /api/v1/knowledge/stats       — 查看知识库统计
POST   /api/v1/knowledge/rebuild     — 重建向量库
POST   /api/v1/knowledge/incremental — 增量索引目录
"""
import os
import logging
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse

from backend.config.settings import get_settings
from backend.models.knowledge import (
    DocumentUploadResponse,
    KnowledgeBaseStats,
    RebuildResponse,
    IncrementalIndexResponse,
    DocumentListResponse,
    DocumentItem,
)
from backend.api.deps import get_rag_pipeline
from backend.rag.page_knowledge import get_page_knowledge_store

logger = logging.getLogger("ai_rd_agent")
router = APIRouter(tags=["知识库管理"])

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
# 最大文件大小：50MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="待上传的文档（txt/pdf/docx）"),
):
    """上传文档并自动索引到知识库

    处理流程：保存文件 → 文本提取 → 切割 → 嵌入 → 写入 Chroma
    """
    # ---- 校验文件名 ----
    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={"error": True, "message": "文件名不能为空"},
        )

    # ---- 校验文件格式 ----
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "message": f"不支持的文件格式 '{ext}'，仅支持: {', '.join(ALLOWED_EXTENSIONS)}",
            },
        )

    # ---- 保存上传文件 ----
    settings = get_settings()
    upload_dir = Path(settings.get_upload_dir())
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 使用临时文件避免直接暴露原始路径
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=ext,
        dir=str(upload_dir),
    ) as tmp_file:
        content = await file.read()

        # 校验文件大小
        if len(content) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "message": f"文件过大（{len(content) / 1024 / 1024:.1f}MB），最大允许 50MB",
                },
            )

        tmp_file.write(content)
        saved_path = tmp_file.name

    # 重命名为原始文件名（防止冲突加序号）
    final_path = upload_dir / file.filename
    if final_path.exists():
        stem = final_path.stem
        counter = 1
        while final_path.exists():
            final_path = upload_dir / f"{stem}_{counter}{ext}"
            counter += 1

    os.rename(saved_path, str(final_path))

    # ---- 索引文档 ----
    try:
        pipeline = get_rag_pipeline()
        chunk_count = pipeline.ingest_file(str(final_path))
        file_size = final_path.stat().st_size
    except Exception as e:
        logger.error(f"文档索引失败: {e}", exc_info=True)
        # 清理已保存的文件
        final_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=500,
            content={"error": True, "message": f"文档索引失败: {str(e)}"},
        )

    return DocumentUploadResponse(
        filename=file.filename,
        file_type=ext.lstrip("."),
        file_size_bytes=file_size,
        chunk_count=chunk_count,
    )


@router.get("/stats", response_model=KnowledgeBaseStats)
async def get_knowledge_stats():
    """获取知识库统计信息"""
    pipeline = get_rag_pipeline()
    stats = pipeline.stats()

    # 统计上传目录中的文档数
    settings = get_settings()
    upload_dir = Path(settings.get_upload_dir())
    doc_count = 0
    if upload_dir.exists():
        doc_count = sum(
            1 for _ in upload_dir.iterdir()
            if _.is_file() and _.suffix.lower() in ALLOWED_EXTENSIONS
        )

    return KnowledgeBaseStats(
        total_documents=doc_count,
        total_chunks=stats.get("total_chunks", 0),
        persist_directory=stats.get("persist_directory", ""),
    )


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_knowledge_base():
    """重建向量库：清空所有向量，重新索引文档目录中的所有文档"""
    try:
        pipeline = get_rag_pipeline()
        chunk_count = pipeline.rebuild()

        return RebuildResponse(
            status="success",
            chunks_created=chunk_count,
            message=f"向量库已重建，共创建 {chunk_count} 个文本块",
        )
    except Exception as e:
        logger.error(f"重建向量库失败: {e}", exc_info=True)
        return RebuildResponse(
            status="failed",
            message=f"重建失败: {str(e)}",
        )


@router.post("/incremental", response_model=IncrementalIndexResponse)
async def incremental_index_knowledge_base():
    """增量索引：只处理新增、修改、删除的文档"""
    try:
        settings = get_settings()
        upload_dir = settings.get_upload_dir()

        pipeline = get_rag_pipeline()
        result = pipeline.ingest_directory_incremental(upload_dir)

        return IncrementalIndexResponse(
            status="success",
            added=result["added"],
            modified=result["modified"],
            removed=result["removed"],
            unchanged=result["unchanged"],
            chunks_created=result["chunks"],
            message=(
                f"增量索引完成：新增 {result['added']} 个，"
                f"修改 {result['modified']} 个，删除 {result['removed']} 个，"
                f"未变 {result['unchanged']} 个，共 {result['chunks']} 个块"
            ),
        )
    except Exception as e:
        logger.error(f"增量索引失败: {e}", exc_info=True)
        return IncrementalIndexResponse(
            status="failed",
            message=f"增量索引失败: {str(e)}",
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_knowledge_docs():
    """获取知识库中已上传文档列表"""
    settings = get_settings()
    upload_dir = Path(settings.get_upload_dir())

    docs = []
    total_chunks = 0
    if upload_dir.exists():
        for f in sorted(upload_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
                docs.append(DocumentItem(
                    filename=f.name,
                    file_size_bytes=f.stat().st_size,
                    chunk_count=0,  # 暂不统计单文档 chunk 数
                    uploaded_at="",
                ))

    try:
        pipeline = get_rag_pipeline()
        stats = pipeline.stats()
        total_chunks = stats.get("total_chunks", 0)
    except Exception:
        pass

    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=total_chunks,
    )


# ==================== 页面知识管理 ====================

@router.get("/pages")
async def list_page_knowledge(limit: int = 100):
    """列出已缓存的页面知识"""
    store = get_page_knowledge_store()
    pages = store.list_pages(limit=limit)
    return {
        "total": len(pages),
        "pages": [p.model_dump() for p in pages],
    }


@router.get("/pages/search")
async def search_page_knowledge(query: str, top_k: int = 5):
    """语义搜索页面知识"""
    store = get_page_knowledge_store()
    pages = store.search(query, top_k=top_k)
    return {
        "query": query,
        "results": [p.model_dump() for p in pages],
    }


@router.delete("/pages")
async def delete_page_knowledge(url: str):
    """删除指定 URL 的页面知识"""
    store = get_page_knowledge_store()
    success = store.delete_page(url)
    if not success:
        return JSONResponse(status_code=404, content={"error": True, "message": "页面知识不存在"})
    return {"status": "success", "message": "已删除"}
