"""
可观测性路由
- GET /api/v1/metrics/tokens: LLM Token 消耗聚合查询（JSON）
（Prometheus 抓取端点 /metrics 在 main.py 注册）
"""
from fastapi import APIRouter

from backend.monitoring.token_tracker import token_tracker

router = APIRouter(tags=["metrics"])


@router.get("/tokens")
async def get_token_usage():
    """LLM Token 消耗统计（进程启动以来累计）"""
    return token_tracker.snapshot()
