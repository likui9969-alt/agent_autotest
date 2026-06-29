"""
FastAPI 中间件模块
提供请求日志记录、耗时统计等功能
"""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from backend.config.json_formatter import request_id_var

logger = logging.getLogger("ai_rd_agent")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件 — 记录每个 HTTP 请求的方法、路径、状态码和耗时"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 执行后续处理链
        response = await call_next(request)

        # 计算耗时
        duration_ms = (time.time() - start_time) * 1000

        # 记录请求信息
        rid = request_id_var.get()
        extras = f" rid={rid}" if rid else ""
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} "
            f"({duration_ms:.1f}ms){extras}"
        )

        # 在响应头中附加耗时信息
        response.headers["X-Process-Time-ms"] = str(round(duration_ms, 1))
        return response
