"""
FastAPI 应用入口
创建应用实例，注册中间件、异常处理器和路由
启动命令：uvicorn backend.main:app --reload
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import get_settings
from backend.config.logging_config import setup_logging
from backend.api.middleware import RequestLoggingMiddleware
from backend.api.exceptions import register_exception_handlers
from backend.api.routes import register_routes

# ---- 初始化配置和日志 ----
settings = get_settings()
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理
    启动时：初始化 Chroma 客户端、预热模型等
    关闭时：清理资源
    """
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} 启动中...")
    logger.info(f"   LLM 模型: {settings.LLM_MODEL}")
    logger.info(f"   嵌入模型: {settings.EMBEDDING_MODEL}")
    logger.info(f"   Chroma 目录: {settings.get_chroma_dir()}")
    yield  # 应用运行期间
    logger.info(f"👋 {settings.APP_NAME} 已关闭")


# ---- 创建 FastAPI 应用实例 ----
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 RAG 的自动化测试与故障分析系统",
    lifespan=lifespan,
)

# ---- CORS 中间件（允许前端跨域访问） ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 请求日志中间件 ----
app.add_middleware(RequestLoggingMiddleware)

# ---- 注册全局异常处理器 ----
register_exception_handlers(app)

# ---- 注册所有业务路由 ----
register_routes(app)


@app.get("/")
async def root():
    """根路径 — 返回项目基本信息"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
