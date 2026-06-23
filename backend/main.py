"""
FastAPI 应用入口
创建应用实例，注册中间件、异常处理器和路由
启动命令：uvicorn backend.main:app --reload --port 8001
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from backend.config.settings import get_settings
from backend.config.logging_config import setup_logging
from backend.api.middleware import RequestLoggingMiddleware
from backend.api.exceptions import register_exception_handlers
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
