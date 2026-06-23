"""
自动化测试路由
POST /api/v1/testing/run — 执行自动化测试并返回报告
"""
import logging
import asyncio
from fastapi import APIRouter, Depends

from backend.models.testing import TestRunRequest, TestReport
from backend.agent.test_executor import TestExecutorAgent
from backend.api.deps import get_test_executor

logger = logging.getLogger("ai_rd_agent")
router = APIRouter(tags=["自动化测试"])


@router.post("/run", response_model=TestReport)
async def run_tests(
    request: TestRunRequest,
    executor: TestExecutorAgent = Depends(get_test_executor),
):
    """执行自动化测试

    支持按场景选择性执行：
    - login：登录流程测试
    - search：搜索流程测试
    - order：下单流程测试

    失败时自动调用 AI 日志分析。

    请求示例：
    {
        "scenarios": ["login", "search"],
        "base_url": "https://example.com",
        "headless": true,
        "timeout_seconds": 30,
        "auto_analyze": true
    }
    """
    # 在线程池中执行，避免同步的 Selenium/mock sleep 阻塞事件循环
    report = await asyncio.to_thread(executor.run_tests, request)

    logger.info(
        f"测试执行完成: {report.passed_count}/{report.total_scenarios} 通过"
    )

    return report
