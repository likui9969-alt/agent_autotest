"""
审计日志中间件
记录关键操作（Agent 执行、RAG 数据变更）到独立的 audit.log 文件，
用于安全审计和操作追溯。
"""
import logging
import json
from datetime import datetime, timezone
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from backend.config.settings import get_settings
from backend.config.json_formatter import request_id_var

# 需要审计的路径前缀
_AUDIT_PREFIXES = (
    "/api/v1/agent/execute",
    "/api/v1/rag/ingest",
    "/api/v1/rag/rebuild",
    "/api/v1/rag/document",
    "/api/v1/knowledge/",
)

logger = logging.getLogger("audit")


class AuditLogMiddleware(BaseHTTPMiddleware):
    """审计日志中间件

    拦截写操作（POST/DELETE/PUT），记录到 audit.log：
    - 时间戳、请求路径、方法
    - 请求体摘要（前 200 字符，JSON 序列化）
    - 响应状态码、耗时
    - API Token 调用者标识（如有）
    """

    async def dispatch(self, request: Request, call_next):
        # 只审计配置的路径
        if not self._should_audit(request):
            return await call_next(request)

        # 读取请求体（可重放，不阻塞后续中间件）
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8", errors="replace")[:200]

        start_time = datetime.now(timezone.utc)
        response = await call_next(request)
        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # 构造审计记录
        settings = get_settings()
        caller = ""
        if settings.API_TOKEN:
            caller = f"token:{settings.API_TOKEN[:4]}***"

        record = {
            "timestamp": start_time.isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(elapsed_ms, 1),
            "request_summary": body_text,
            "request_id": request_id_var.get(),
        }
        if caller:
            record["caller"] = caller

        # 写入 audit.log（JSON 一行一条）
        logger.info(json.dumps(record, ensure_ascii=False))

        return response

    def _should_audit(self, request: Request) -> bool:
        """判断请求是否需要审计"""
        if request.method not in ("POST", "PUT", "DELETE", "PATCH"):
            return False
        path = request.url.path
        return any(path.startswith(prefix) for prefix in _AUDIT_PREFIXES)
