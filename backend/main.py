"""
FastAPI 应用入口
创建应用实例，注册中间件、异常处理器和路由
启动命令：uvicorn backend.main:app --reload --port 8001
"""
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from backend.config.settings import get_settings
from backend.config.logging_config import setup_logging
from backend.config.json_formatter import request_id_var
from backend.api.middleware import RequestLoggingMiddleware
from backend.api.audit import AuditLogMiddleware
from backend.api.exceptions import register_exception_handlers
from backend.api.auth import AuthMiddleware
from backend.api.routes import register_routes

# ---- 初始化配置（使用标准 logger，lifespan 中替换为配置版本） ----
settings = get_settings()
logger = logging.getLogger("ai_rd_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理
    启动时：初始化日志、Chroma 客户端、预热模型等
    关闭时：清理资源
    """
    global logger
    logger = setup_logging()  # 应用启动时才执行，而非模块导入时
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} 启动中...")
    logger.info(f"   LLM 模型: {settings.LLM_MODEL}")
    logger.info(f"   嵌入模型: {settings.EMBEDDING_MODEL}")
    logger.info(f"   Chroma 目录: {settings.get_chroma_dir()}")

    # 预热 RAG 向量库（触发 Chroma PersistentClient 初始化，避免首次请求等待）
    try:
        from backend.api.deps import get_rag_pipeline
        pipeline = get_rag_pipeline()
        stats = pipeline.stats()
        logger.info(f"RAG 向量库预热完成: {stats.get('total_chunks', 0)} 个块")
    except Exception as e:
        logger.warning(f"RAG 预热失败（首次请求时将自动初始化）: {e}")

    yield  # 应用运行期间
    logger.info(f"👋 {settings.APP_NAME} 已关闭")


# ---- 创建 FastAPI 应用实例 ----
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 RAG 的自动化测试与故障分析系统",
    lifespan=lifespan,
)

# ---- CORS 中间件（允许前端跨域访问，从配置读取允许源） ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 请求 ID 中间件（早于日志中间件，确保 request_id 已设置） ----
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """为每个请求分配唯一 request_id，注入 contextvars 供日志使用"""
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

# ---- 请求日志中间件 ----
app.add_middleware(RequestLoggingMiddleware)

# ---- 速率限制中间件 ----
if settings.RATE_LIMIT_ENABLED:
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.errors import RateLimitExceeded

        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[settings.RATE_LIMIT_DEFAULT],
            enabled=True,
        )
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)
        logger.info(f"速率限制已启用: 默认 {settings.RATE_LIMIT_DEFAULT}")
    except ImportError:
        logger.warning("slowapi 未安装，速率限制已跳过 (pip install slowapi)")

# ---- API 认证中间件（可选，API_TOKEN 配置时启用） ----
app.add_middleware(AuthMiddleware)

# ---- 审计日志中间件（记录关键操作到 audit.log） ----
app.add_middleware(AuditLogMiddleware)

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
        "demo": "/demo",
    }


# ==================== Demo 测试站点 ====================
# 内置的演示页面，供 Selenium 自动化测试使用
# 包含登录表单、搜索框、下单表单，元素 ID 与测试场景定位器匹配

_DEMO_HTML_PATH = Path(__file__).parent / "static" / "demo.html"


@app.get("/demo", response_class=HTMLResponse)
@app.get("/demo/", response_class=HTMLResponse)
@app.get("/demo/login", response_class=HTMLResponse)
@app.get("/demo/order", response_class=HTMLResponse)
async def demo_page():
    """Demo 测试站点 — 提供 Selenium 自动化测试所需的目标页面

    - /demo        首页（含搜索 + 登录 + 下单）
    - /demo/login  登录页
    - /demo/order  下单页
    """
    return HTMLResponse(content=_DEMO_HTML_PATH.read_text(encoding="utf-8"))
