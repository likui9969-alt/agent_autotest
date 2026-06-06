"""
RAG 智能问答路由
POST /api/v1/rag/query — RAG 检索增强生成问答
"""
import logging
from fastapi import APIRouter, Depends

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
    # 校验 search_type
    if request.search_type not in ("similarity", "mmr"):
        from fastapi.responses import JSONResponse
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
