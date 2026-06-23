"""
健康检查路由
GET /health — 返回服务及依赖组件的存活状态
"""
import logging
from fastapi import APIRouter

logger = logging.getLogger("ai_rd_agent")

router = APIRouter(tags=["系统"])


@router.get("/health")
async def health_check():
    """健康检查接口 — 验证服务和关键依赖是否正常"""
    health_status = {
        "status": "ok",
        "service": "AI研发效能智能体",
    }

    try:
        from backend.config.settings import get_settings
        settings = get_settings()
        health_status["llm_model"] = settings.LLM_MODEL
        health_status["embedding_model"] = settings.EMBEDDING_MODEL
    except Exception as e:
        logger.warning(f"健康检查读取配置失败: {e}")
        health_status["status"] = "error"
        health_status["errors"] = [f"config: {e}"]
        return health_status

    # 检测 LLM 可用性（轻量嵌入调用）
    try:
        from backend.api.deps import get_llm_client
        llm_client = get_llm_client()
        llm_client.embed_single("health_check")
        health_status["llm"] = "ok"
    except Exception as e:
        logger.warning(f"LLM 健康检查失败: {e}")
        health_status["llm"] = "error"
        health_status.setdefault("errors", []).append(f"llm: {e}")
        health_status["status"] = "degraded"

    # 检测 Chroma 可用性
    try:
        from backend.api.deps import get_rag_pipeline
        pipeline = get_rag_pipeline()
        count = pipeline.vector_store.count()
        health_status["chroma"] = "ok"
        health_status["vector_chunks"] = count
    except Exception as e:
        logger.warning(f"Chroma 健康检查失败: {e}")
        health_status["chroma"] = "error"
        health_status.setdefault("errors", []).append(f"chroma: {e}")
        health_status["status"] = "degraded"

    return health_status
