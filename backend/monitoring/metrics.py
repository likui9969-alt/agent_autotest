"""
Prometheus 指标定义与 HTTP 指标中间件
====================================

指标清单：
- http_requests_total{method, path, status}          HTTP 请求计数
- http_request_duration_seconds{method, path}        HTTP 请求耗时直方图
- llm_calls_total{provider, operation, status}       LLM 调用计数（含重试次数）
- llm_call_duration_seconds{provider, operation}     LLM 调用耗时直方图
- llm_tokens_total{provider, operation, type}        LLM Token 消耗（type=prompt/completion）

暴露方式：GET /metrics（Prometheus 文本协议），供 Prometheus 抓取。

标签基数控制：path 使用路由模板（如 /api/v1/rag/query），
未匹配路由的请求归入 "unmatched"，避免 404 路径撑爆标签基数。
"""
import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ==================== HTTP 指标 ====================

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "HTTP 请求总数",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求耗时（秒）",
    ["method", "path"],
)

# ==================== LLM 指标 ====================

LLM_CALLS_TOTAL = Counter(
    "llm_calls_total",
    "LLM 调用总次数（含重试，按 attempt 计）",
    ["provider", "operation", "status"],
)

LLM_CALL_DURATION = Histogram(
    "llm_call_duration_seconds",
    "LLM 调用耗时（秒）",
    ["provider", "operation"],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "LLM Token 消耗总数",
    ["provider", "operation", "type"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """HTTP 指标中间件 — 记录每个请求的计数与耗时

    path 标签使用路由模板；异常请求按 500 记录后重新抛出。
    """

    async def dispatch(self, request: Request, call_next):
        method = request.method
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            HTTP_REQUESTS_TOTAL.labels(method, self._path_label(request), 500).inc()
            HTTP_REQUEST_DURATION.labels(method, self._path_label(request)).observe(
                time.perf_counter() - start
            )
            raise

        HTTP_REQUESTS_TOTAL.labels(method, self._path_label(request), response.status_code).inc()
        HTTP_REQUEST_DURATION.labels(method, self._path_label(request)).observe(
            time.perf_counter() - start
        )
        return response

    @staticmethod
    def _path_label(request: Request) -> str:
        """路由模板作为 path 标签（基数受控）；未匹配路由归入 unmatched"""
        route = request.scope.get("route")
        if route is not None and getattr(route, "path", None):
            return route.path
        return "unmatched"


def metrics_response() -> Response:
    """构造 /metrics 端点响应（Prometheus 文本协议）"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
