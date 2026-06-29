"""
自动化测试路由
POST /api/v1/testing/run — 执行自动化测试并返回报告
POST /api/v1/testing/run/async — 异步执行测试（返回 task_id）
GET  /api/v1/testing/tasks/{task_id} — 查询异步任务状态
POST /api/v1/testing/tasks/{task_id}/cancel — 取消异步测试任务
"""
import logging
import asyncio
import threading
import uuid
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from backend.models.testing import TestRunRequest, TestReport
from backend.agent.test_executor import TestExecutorAgent
from backend.api.deps import get_test_executor
from backend.db.reports import TestReportStore

logger = logging.getLogger("ai_rd_agent")
router = APIRouter(tags=["自动化测试"])

# 测试报告存储
_report_store = TestReportStore()

# 内存异步任务状态（单进程有效；多 worker 需改用 Redis/Celery）
_async_tasks: dict[str, dict[str, Any]] = {}

# 任务取消事件（与 _async_tasks 生命周期一致）
_cancel_events: dict[str, threading.Event] = {}


def _run_test_task(task_id: str, request: TestRunRequest, cancel_event: threading.Event) -> None:
    """后台执行测试任务"""
    try:
        executor = TestExecutorAgent()
        report = executor.run_tests(request, cancel_event=cancel_event)

        # 持久化测试报告
        try:
            _report_store.save(
                name=request.scenarios[0] if request.scenarios else "unknown",
                report_type="test",
                status="passed" if report.passed_count == report.total_scenarios else "failed",
                result=report.model_dump(),
            )
        except Exception as e:
            logger.warning(f"保存测试报告失败: {e}")

        # 如果任务被取消，最终状态为 cancelled
        if cancel_event.is_set():
            _async_tasks[task_id] = {
                "status": "cancelled",
                "result": report.model_dump(),
                "error": "任务已取消",
            }
            logger.info(f"[AsyncTask {task_id}] 测试任务已取消")
        else:
            _async_tasks[task_id] = {
                "status": "completed",
                "result": report.model_dump(),
                "error": "",
            }
            logger.info(
                f"[AsyncTask {task_id}] 测试执行完成: "
                f"{report.passed_count}/{report.total_scenarios} 通过"
            )
    except Exception as e:
        logger.error(f"[AsyncTask {task_id}] 测试执行失败: {e}", exc_info=True)
        _async_tasks[task_id] = {
            "status": "failed",
            "result": None,
            "error": str(e)[:500],
        }
    finally:
        # 任务结束后清理取消事件，避免内存泄漏
        _cancel_events.pop(task_id, None)


@router.post("/run/async")
async def run_tests_async(
    request: TestRunRequest,
    background_tasks: BackgroundTasks,
):
    """异步执行自动化测试

    返回 task_id，前端通过 /tasks/{task_id} 轮询结果。
    """
    task_id = str(uuid.uuid4())[:8]
    _async_tasks[task_id] = {"status": "running", "result": None, "error": ""}
    _cancel_events[task_id] = threading.Event()
    background_tasks.add_task(
        _run_test_task, task_id, request, _cancel_events[task_id]
    )
    logger.info(f"[AsyncTask {task_id}] 已创建测试任务")
    return {"task_id": task_id, "status": "running"}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """查询异步测试任务状态"""
    task = _async_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@router.post("/tasks/{task_id}/cancel")
async def cancel_test_task(task_id: str):
    """取消正在执行的异步测试任务"""
    task = _async_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] not in ("running", "pending"):
        return {"task_id": task_id, "cancelled": False, "message": "任务已结束"}

    event = _cancel_events.get(task_id)
    if event is None:
        raise HTTPException(status_code=404, detail="任务取消事件不存在")

    event.set()
    task["status"] = "cancelling"
    logger.info(f"[AsyncTask {task_id}] 收到取消请求")
    return {"task_id": task_id, "cancelled": True, "message": "取消信号已发送"}


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

    # 持久化测试报告
    try:
        _report_store.save(
            name=request.scenarios[0] if request.scenarios else "unknown",
            report_type="test",
            status="passed" if report.passed_count == report.total_scenarios else "failed",
            result=report.model_dump(),
        )
    except Exception as e:
        logger.warning(f"保存测试报告失败: {e}")

    logger.info(
        f"测试执行完成: {report.passed_count}/{report.total_scenarios} 通过"
    )

    return report


@router.get("/reports")
async def list_test_reports(limit: int = 50):
    """查询测试报告列表"""
    return {"reports": _report_store.list_reports(limit=limit, report_type="test")}


@router.get("/reports/{report_id}")
async def get_test_report(report_id: str):
    """获取测试报告详情"""
    report = _report_store.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    return report
