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

    # TODO: 后续步骤中补充 Chroma、LLM 的连接检测
    try:
        from backend.config.settings import get_settings
        settings = get_settings()
        health_status["llm_model"] = settings.LLM_MODEL
        health_status["embedding_model"] = settings.EMBEDDING_MODEL
    except Exception as e:
        logger.warning(f"健康检查部分失败: {e}")

    return health_status
