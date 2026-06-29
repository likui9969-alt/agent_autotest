"""
API 认证模块
提供可选的 Token 认证中间件。
当配置了 API_TOKEN 时自动启用，未配置时跳过认证（向后兼容）。
"""
import logging
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from backend.config.settings import get_settings

logger = logging.getLogger("ai_rd_agent")

# 不需要认证的路径前缀
_SKIP_AUTH_PREFIXES = (
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/demo",
    "/",
)


class AuthMiddleware(BaseHTTPMiddleware):
    """API 认证中间件 — Bearer Token 验证（可选）

    当 settings.API_TOKEN 配置时，除健康检查和文档路径外的所有请求
    需要提供 Authorization: Bearer <token> 头。
    未配置时跳过认证，保持向后兼容。
    """

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        # 未配置 API_TOKEN 或路径在白名单中，跳过认证
        if not settings.API_TOKEN or request.url.path.startswith(_SKIP_AUTH_PREFIXES):
            return await call_next(request)

        # 检查 Authorization 头
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            logger.warning(f"缺少认证 Token: {request.method} {request.url.path}")
            return _unauthorized_response("缺少认证 Token。请在请求头中添加 Authorization: Bearer <token>")

        token = auth_header[7:]  # 去掉 "Bearer " 前缀
        if token != settings.API_TOKEN:
            logger.warning(f"无效的认证 Token: {request.method} {request.url.path}")
            return _unauthorized_response("Token 无效")

        return await call_next(request)


def _unauthorized_response(detail: str):
    """构造 401 响应"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": True, "message": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )
