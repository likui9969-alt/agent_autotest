"""
可观测性模块
- metrics.py: Prometheus 指标定义 + HTTP 指标中间件
- token_tracker.py: LLM Token 消耗追踪（进程内聚合 + 指标上报）
"""
from backend.monitoring.metrics import (
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    LLM_CALL_DURATION,
    LLM_CALLS_TOTAL,
    LLM_TOKENS_TOTAL,
    MetricsMiddleware,
    metrics_response,
)
from backend.monitoring.token_tracker import token_tracker

__all__ = [
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION",
    "LLM_CALLS_TOTAL",
    "LLM_CALL_DURATION",
    "LLM_TOKENS_TOTAL",
    "MetricsMiddleware",
    "metrics_response",
    "token_tracker",
]
