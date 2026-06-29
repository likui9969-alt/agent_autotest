"""
Agent 任务注册表

管理正在运行的 Agent 任务，支持用户主动取消。
每个任务注册一个 asyncio.Event，ReAct 循环中定期检查是否被取消。
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("ai_rd_agent")

# task_id -> asyncio.Event
_tasks: dict[str, asyncio.Event] = {}


def register(task_id: str) -> asyncio.Event:
    """注册一个任务，返回用于取消的 Event"""
    event = asyncio.Event()
    _tasks[task_id] = event
    logger.info(f"[TaskRegistry] 任务注册: {task_id}")
    return event


def cancel(task_id: str) -> bool:
    """取消指定任务

    Returns:
        True: 任务存在并已触发取消
        False: 任务不存在或已完成
    """
    event = _tasks.get(task_id)
    if event is None:
        logger.warning(f"[TaskRegistry] 取消失败，任务不存在: {task_id}")
        return False
    event.set()
    logger.info(f"[TaskRegistry] 任务取消信号已发送: {task_id}")
    return True


def is_cancelled(task_id: str) -> bool:
    """检查任务是否已被取消"""
    event = _tasks.get(task_id)
    return event.is_set() if event else False


def unregister(task_id: str) -> None:
    """任务完成后清理注册信息"""
    _tasks.pop(task_id, None)
    logger.info(f"[TaskRegistry] 任务注销: {task_id}")


def get_event(task_id: str) -> Optional[asyncio.Event]:
    """获取任务的取消 Event（用于在图中检查）"""
    return _tasks.get(task_id)
