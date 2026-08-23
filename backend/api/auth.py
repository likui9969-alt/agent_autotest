"""
API 认证模块
Token 认证中间件，采用 fail-closed 策略：
- 配置了 API_TOKEN：除白名单路径外所有请求需 Bearer Token
- 未配置 API_TOKEN 且 DEBUG=False（生产模式）：拒绝写操作（GET 只读放行）
- 未配置 API_TOKEN 且 DEBUG=True（开发模式）：放行，启动时记录警告
"""
import logging
from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from backend.config.settings import get_settings

logger = logging.getLogger("ai_rd_agent")

# 不需要认证的路径前缀
# 注意：不能用 "/" 作为前缀——所有路径都以 "/" 开头，会放行全部请求
_SKIP_AUTH_PREFIXES = (
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/demo",
)

# 不需要认证的精确路径
_SKIP_AUTH_EXACT = frozenset({"/"})

# 无需认证即可执行的安全方法（只读）
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# 敏感只读路径前缀：生产模式未配置 token 时同样拒绝。
# 这些端点返回对话历史/执行报告/知识库文档内容，不应匿名可读。
_SENSITIVE_READ_PREFIXES = (
    "/api/v1/agent/memory",       # 多轮对话历史
    "/api/v1/agent/reports",      # Agent 执行报告（可能含日志内容）
    "/api/v1/knowledge/documents",  # 知识库文档清单与内容
)


class AuthMiddleware(BaseHTTPMiddleware):
    """API 认证中间件 — Bearer Token 验证（fail-closed）

    认证策略：
    1. 白名单路径（健康检查/文档）直接放行
    2. 配置了 API_TOKEN：所有请求需携带 Authorization: Bearer <token>
    3. 未配置 token 且生产模式（DEBUG=False）：拒绝写操作与敏感只读路径
       （对话历史/执行报告/知识库文档），其余 GET 只读放行
    4. 未配置 token 且开发模式（DEBUG=True）：放行（仅限本地环境）
    """

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        # 白名单路径直接放行
        # 注意："/" 不能放进 startswith 前缀（会匹配所有路径），
        # 根路径用精确匹配
        if request.url.path in _SKIP_AUTH_EXACT or request.url.path.startswith(_SKIP_AUTH_PREFIXES):
            return await call_next(request)

        if settings.API_TOKEN:
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

        # ---- 未配置 API_TOKEN ----
        # 生产模式：写操作 fail-closed；敏感只读路径（对话历史/报告）同样拒绝
        if not settings.DEBUG:
            path = request.url.path
            is_sensitive_read = (
                path.startswith(_SENSITIVE_READ_PREFIXES)
            )
            if request.method not in _SAFE_METHODS or is_sensitive_read:
                logger.error(
                    f"生产模式未配置 API_TOKEN，拒绝请求: {request.method} {path}"
                )
                return _service_unavailable_response(
                    "服务未配置认证（API_TOKEN 为空），写操作与敏感数据接口已被拒绝。"
                    "请在 .env 中设置 API_TOKEN 后重启服务。"
                )

        # 开发模式：放行（仅限本地调试环境）
        return await call_next(request)


def _unauthorized_response(detail: str):
    """构造 401 响应"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": True, "message": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )


def _service_unavailable_response(detail: str):
    """构造 503 响应（认证未配置，服务不可用）"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": True, "message": detail},
    )
