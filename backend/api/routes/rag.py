"""
RAG 智能问答和知识库管理路由
POST /api/v1/rag/query               — RAG 检索增强生成问答
POST /api/v1/rag/ingest              — 上传文件并索引到知识库
POST /api/v1/rag/ingest/directory    — 索引整个目录
POST /api/v1/rag/rebuild             — 重建向量库
GET  /api/v1/rag/stats               — 知识库统计信息
GET  /api/v1/rag/documents           — 列出知识库所有文档
DELETE /api/v1/rag/document           — 按文件名删除文档
"""
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from backend.models.rag import RAGQueryRequest, RAGQueryResponse
from backend.api.deps import get_rag_pipeline

logger = logging.getLogger("ai_rd_agent")
router = APIRouter(tags=["智能问答"])


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """RAG 智能问答接口

    接收用户问题，检索知识库中的相关内容，交由 LLM 生成回答。

    请求示例：
    {
        "question": "登录接口返回500错误怎么办",
        "top_k": 5,
        "search_type": "similarity",
        "include_sources": true
    }
    """
    if request.search_type not in ("similarity", "mmr"):
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "message": "search_type 仅支持 'similarity' 或 'mmr'",
            },
        )

    pipeline = get_rag_pipeline()
    response = pipeline.query(request)

    logger.info(
        f"RAG 查询完成 | 问题: {request.question[:50]}... | "
        f"检索到 {response.retrieved_count} 个文档 | "
        f"耗时 {response.response_time_ms:.0f}ms"
    )

    return response


@router.post("/ingest")
async def rag_ingest(
    file: UploadFile = File(..., description="待索引的文档文件"),
):
    """上传文件并索引到知识库

    支持格式：.txt / .md / .csv / .pdf / .docx（最大 10MB）

    请求示例：
        curl -X POST http://localhost:8000/api/v1/rag/ingest \\
            -F "file=@report.txt"
    """
    pipeline = get_rag_pipeline()

    # 保存上传到临时文件
    from backend.config.settings import get_settings
    settings = get_settings()
    upload_dir = Path(settings.get_upload_dir())
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "upload").suffix.lower()
    safe_name = f"upload_{len(list(upload_dir.iterdir()))}{ext}"
    save_path = upload_dir / safe_name

    content = await file.read()
    save_path.write_bytes(content)

    logger.info(f"文件已保存: {save_path} ({len(content)} bytes)")

    try:
        chunk_count = pipeline.ingest_file(str(save_path))
        return {
            "status": "success",
            "filename": file.filename,
            "saved_as": str(save_path),
            "chunks_created": chunk_count,
        }
    except Exception as e:
        logger.error(f"文件索引失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": f"文件索引失败: {str(e)}",
            },
        )


@router.post("/ingest/directory")
async def rag_ingest_directory(
    dir_path: str = Form(..., description="文档目录路径（相对于项目根目录）"),
    batch_size: int = Form(10, description="每批处理文件数（流式索引），设为 0 一次性加载"),
):
    """索引指定目录下的所有文档

    使用流式处理，每 batch_size 个文件一批，避免大目录内存溢出。

    请求示例：
    {
        "dir_path": "data/docs",
        "batch_size": 10
    }
    """
    from backend.config.settings import PROJECT_ROOT

    target_dir = PROJECT_ROOT / dir_path
    if not target_dir.exists() or not target_dir.is_dir():
        return JSONResponse(
            status_code=404,
            content={
                "error": True,
                "message": f"目录不存在: {target_dir}",
            },
        )

    try:
        pipeline = get_rag_pipeline()
        total_chunks = pipeline.ingest_directory_batch(str(target_dir), batch_size=batch_size)
        return {
            "status": "success",
            "directory": str(target_dir),
            "batch_size": batch_size,
            "total_chunks": total_chunks,
        }
    except Exception as e:
        logger.error(f"目录索引失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": f"目录索引失败: {str(e)}",
            },
        )


@router.post("/rebuild")
async def rag_rebuild(
    dir_path: str = Form(default="", description="文档目录路径（留空使用配置的默认目录）"),
):
    """重建向量库：删除所有索引数据，从指定目录重新索引

    请求示例：
    {
        "dir_path": "data/docs"
    }
    """
    try:
        pipeline = get_rag_pipeline()
        target = dir_path or None
        total_chunks = pipeline.rebuild(target)
        return {
            "status": "success",
            "chunks_indexed": total_chunks,
        }
    except Exception as e:
        logger.error(f"重建向量库失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": f"重建失败: {str(e)}",
            },
        )


@router.get("/stats")
async def rag_stats():
    """获取知识库统计信息"""
    try:
        pipeline = get_rag_pipeline()
        stats = pipeline.stats()
        return stats
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": f"获取统计信息失败: {str(e)}",
            },
        )


@router.get("/documents")
async def rag_list_documents():
    """列出知识库中所有文档及其块数"""
    try:
        pipeline = get_rag_pipeline()
        docs = pipeline.get_documents()
        return {
            "status": "success",
            "total": len(docs),
            "documents": docs,
        }
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": True, "message": f"获取文档列表失败: {str(e)}"},
        )


@router.delete("/document")
async def rag_delete_document(filename: str = Form(..., description="要删除的文档文件名")):
    """按文件名从知识库中删除文档

    请求示例：
        curl -X DELETE http://localhost:8000/api/v1/rag/document \\
            -d "filename=report.txt"
    """
    try:
        pipeline = get_rag_pipeline()
        deleted = pipeline.delete_document(filename)
        if deleted == 0:
            return JSONResponse(
                status_code=404,
                content={
                    "error": True,
                    "message": f"未找到文档: {filename}",
                },
            )
        return {
            "status": "success",
            "filename": filename,
            "chunks_deleted": deleted,
        }
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": True, "message": f"删除文档失败: {str(e)}"},
        )
