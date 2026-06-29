"""
Agent 执行路由 — LangGraph Supervisor + ReAct 循环
POST /api/v1/agent/execute      — 执行 Agent 任务（异步）
POST /api/v1/agent/execute/stream — 执行 Agent 任务（SSE 流式，实时展示推理过程）
"""
import logging
import json
import uuid
import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agent.graph import get_supervisor_graph
from backend.agent.state import AgentState
from backend.agent.memory import (
    SessionMemoryManager,
    get_conversation_memory_store,
)
from backend.agent.task_registry import register, cancel, unregister, is_cancelled
from backend.config.settings import get_settings
from backend.db.reports import TestReportStore

logger = logging.getLogger("ai_rd_agent")
router = APIRouter(tags=["Agent执行"])

# 全局会话记忆管理器（内存 LRU 缓存）
_session_manager = SessionMemoryManager(max_sessions=100)

# 持久化对话记忆存储
_memory_store = get_conversation_memory_store()

# Agent 执行报告存储
_report_store = TestReportStore()


# ==================== 请求模型 ====================

class AgentExecuteRequest(BaseModel):
    """Agent 执行请求"""
    task: str = Field(..., description="任务描述（自然语言）", min_length=1, max_length=2000)
    task_type: str = Field(
        default="auto",
        description="任务类型：auto（自动识别）/ rag_query / log_analysis / test_execution / jira_create"
    )
    max_iterations: int = Field(default=5, ge=1, le=10, description="ReAct 最大迭代次数")
    stream: bool = Field(default=False, description="是否使用 SSE 流式返回")
    session_id: str = Field(default="", description="会话标识（用于多轮对话记忆，留空则不启用记忆）")


class AgentExecuteResponse(BaseModel):
    """Agent 执行响应"""
    task_id: str
    task_type: str
    final_response: str
    tool_calls_made: int
    iterations: int
    execution_time_ms: float
    total_tokens: int = 0
    token_usage: list[dict] = []
    error: str = ""


# ==================== 核心执行逻辑 ====================

def _execute_agent(
    task: str,
    max_iterations: int,
    session_id: str = "",
    memory_context: str = "",
    task_id: str = "",
) -> dict:
    """执行 Agent 任务的核心逻辑

    Args:
        task: 用户任务描述
        max_iterations: ReAct 最大迭代次数
        session_id: 会话标识
        memory_context: 记忆上下文
        task_id: 任务标识（用于取消）

    Returns:
        执行结果字典
    """
    graph = get_supervisor_graph()

    initial_state: AgentState = {
        "messages": [],
        "task_type": "unknown",
        "user_input": task,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "tool_calls": [],
        "tool_results": [],
        "tool_history": [],
        "token_usage": [],
        "context": {},
        "final_response": "",
        "next_action": "",
        "error": "",
        "session_id": session_id,
        "memory_context": memory_context,
        "task_id": task_id,
    }

    # 执行图
    recursion_limit = get_settings().AGENT_RECURSION_LIMIT
    result = graph.invoke(initial_state, {"recursion_limit": recursion_limit})

    # 提取结果
    tool_calls_made = len(result.get("tool_results", []))
    iterations = result.get("iteration_count", 0)
    token_usage_list = result.get("token_usage", [])
    total_tokens = sum(u.get("total_tokens", 0) for u in token_usage_list)

    return {
        "task_type": result.get("task_type", "unknown"),
        "final_response": result.get("final_response", ""),
        "tool_calls_made": tool_calls_made,
        "iterations": iterations,
        "error": result.get("error", ""),
        "messages": result.get("messages", []),
        "tool_results": result.get("tool_results", []),
        "token_usage": token_usage_list,
        "total_tokens": total_tokens,
        "session_id": session_id,
    }


# ==================== 异步执行端点 ====================

@router.post("/execute", response_model=AgentExecuteResponse)
async def execute_agent(request: AgentExecuteRequest):
    """执行 Agent 任务（异步，返回完整结果）

    请求示例：
    {
        "task": "分析一下登录超时错误，看看有没有历史案例",
        "max_iterations": 5
    }
    """
    task_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()
    settings = get_settings()
    timeout = request.max_iterations * 30 if request.task_type == "auto" else settings.AGENT_TIMEOUT_SECONDS

    # 注册任务，支持取消
    register(task_id)

    # 多轮对话记忆（优先使用持久化存储）
    memory_context = ""
    if request.session_id:
        memory_context = _memory_store.format_context(request.session_id, limit=5)
        turn_count = _memory_store.count_turns(request.session_id)
        logger.info(f"[Agent {task_id}] 会话 {request.session_id}: {turn_count} 轮历史")

    logger.info(f"[Agent API {task_id}] 收到任务: {request.task[:100]}...")

    try:
        # 在线程池中执行（避免阻塞事件循环），带超时控制
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _execute_agent,
                request.task,
                request.max_iterations,
                request.session_id,
                memory_context,
                task_id,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(f"[Agent API {task_id}] 执行超时 (>{timeout}s)")
        return AgentExecuteResponse(
            task_id=task_id,
            task_type="timeout",
            final_response=f"Agent 执行超时（超过 {timeout} 秒）。请尝试简化问题描述或增加 max_iterations。",
            tool_calls_made=0,
            iterations=0,
            execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            error=f"执行超时 ({timeout}s)",
        )
    except Exception as e:
        logger.error(f"[Agent API {task_id}] 执行失败: {e}", exc_info=True)
        return AgentExecuteResponse(
            task_id=task_id,
            task_type="error",
            final_response=f"Agent 执行失败: {str(e)}",
            tool_calls_made=0,
            iterations=0,
            execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            error=str(e),
        )
    finally:
        unregister(task_id)

    elapsed = (datetime.now() - start_time).total_seconds() * 1000

    response = AgentExecuteResponse(
        task_id=task_id,
        task_type=result["task_type"],
        final_response=result["final_response"],
        tool_calls_made=result["tool_calls_made"],
        iterations=result["iterations"],
        execution_time_ms=round(elapsed, 1),
        error=result["error"],
    )

    logger.info(
        f"[Agent API {task_id}] 完成: "
        f"type={response.task_type}, "
        f"tools={response.tool_calls_made}, "
        f"iters={response.iterations}, "
        f"time={response.execution_time_ms:.0f}ms"
    )

    # 持久化 Agent 执行报告
    try:
        _report_store.save(
            name=request.task[:50],
            report_type="agent",
            status="error" if response.error else "passed",
            result=response.model_dump(),
            report_id=task_id,
        )
    except Exception as e:
        logger.warning(f"保存 Agent 报告失败: {e}")

    # 持久化本轮对话（用户输入 + Agent 回答）
    if request.session_id:
        try:
            _memory_store.add_turn(
                request.session_id,
                request.task,
                response.final_response,
            )
            # 同步更新内存缓存，保持两者一致
            mem = _session_manager.get_or_create(request.session_id)
            mem.add_turn(request.task, response.final_response)
        except Exception as e:
            logger.warning(f"保存对话记忆失败: {e}")

    return response


# ==================== SSE 流式端点 ====================

@router.post("/execute/stream")
async def execute_agent_stream(request: AgentExecuteRequest):
    """执行 Agent 任务（SSE 流式，实时展示推理过程）

    使用 LangGraph 的 stream() 方法，每个图节点执行完毕后立即推送事件。

    客户端通过 EventSource 接收以下事件类型：
    - start: 任务开始
    - node_start: 进入节点 (supervisor / rag_node / ...)
    - node_end: 节点执行完毕
    - tool_call: Agent 调用工具
    - final: 最终回答
    - error: 错误信息
    - done: SSE 流结束
    """

    async def event_generator():
        task_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()

        # 注册任务，支持取消
        register(task_id)

        # 加载持久化记忆上下文
        memory_context = ""
        final_response = ""
        if request.session_id:
            memory_context = _memory_store.format_context(request.session_id, limit=5)

        try:
            # 发送开始事件
            yield _sse_event("start", {
                "task_id": task_id,
                "task": request.task[:200],
                "max_iterations": request.max_iterations,
            })

            # 构建初始状态
            graph = get_supervisor_graph()
            initial_state: AgentState = {
                "messages": [],
                "task_type": "unknown",
                "user_input": request.task,
                "iteration_count": 0,
                "max_iterations": request.max_iterations,
                "tool_calls": [],
                "tool_results": [],
                "tool_history": [],
                "token_usage": [],
                "context": {},
                "final_response": "",
                "next_action": "",
                "error": "",
                "session_id": request.session_id,
                "memory_context": memory_context,
                "task_id": task_id,
            }

            # 使用 stream() 逐节点推送
            tool_count = 0
            iter_count = 0
            total_tokens = 0

            for event in graph.stream(initial_state, {"recursion_limit": 50}):
                # 检查是否被取消
                if is_cancelled(task_id):
                    logger.info(f"[Agent SSE {task_id}] 任务已取消，终止流")
                    yield _sse_event("error", {"message": "任务已被用户取消", "cancelled": True})
                    break

                # event 格式: {node_name: state_update}
                for node_name, state_update in event.items():
                    yield _sse_event("node_start", {
                        "node": node_name,
                        "iteration": state_update.get("iteration_count", 0),
                    })

                    # 记录工具调用
                    tool_calls = state_update.get("tool_calls", [])
                    if tool_calls:
                        tool_count += len(tool_calls)
                        for tc in tool_calls:
                            yield _sse_event("tool_call", {
                                "tool": tc.get("name", "unknown"),
                                "args": {k: str(v)[:100] for k, v in tc.get("args", {}).items()},
                            })

                    # 记录最终输出
                    resp = state_update.get("final_response", "")
                    if resp:
                        final_response = resp

                    iter_count = max(iter_count, state_update.get("iteration_count", 0))

                    # 累加 token 消耗
                    for usage in state_update.get("token_usage", []):
                        total_tokens += usage.get("total_tokens", 0)

                    yield _sse_event("node_end", {
                        "node": node_name,
                        "next_action": state_update.get("next_action", ""),
                    })

            # 发送最终结果
            if not is_cancelled(task_id):
                yield _sse_event("final", {
                    "task_type": "completed",
                    "response": final_response,
                    "tool_calls": tool_count,
                    "iterations": iter_count,
                    "total_tokens": total_tokens,
                    "time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                })

        except asyncio.TimeoutError:
            logger.error(f"[Agent SSE {task_id}] 执行超时")
            yield _sse_event("error", {"message": f"执行超时（超过设置时间）"})

        except Exception as e:
            logger.error(f"[Agent SSE {task_id}] 错误: {e}", exc_info=True)
            yield _sse_event("error", {"message": str(e)})

        finally:
            unregister(task_id)
            # 持久化本轮对话
            if request.session_id and final_response:
                try:
                    _memory_store.add_turn(
                        request.session_id,
                        request.task,
                        final_response,
                    )
                    mem = _session_manager.get_or_create(request.session_id)
                    mem.add_turn(request.task, final_response)
                except Exception as e:
                    logger.warning(f"SSE 保存对话记忆失败: {e}")

        yield _sse_event("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event_type: str, data: dict) -> str:
    """构建 SSE 格式的事件字符串"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==================== 取消任务端点 ====================

class AgentCancelRequest(BaseModel):
    """取消 Agent 任务请求"""
    task_id: str = Field(..., description="要取消的任务 ID")


class AgentCancelResponse(BaseModel):
    """取消 Agent 任务响应"""
    task_id: str
    cancelled: bool
    message: str


@router.post("/cancel", response_model=AgentCancelResponse)
async def cancel_agent_task(request: AgentCancelRequest):
    """取消正在执行的 Agent 任务"""
    success = cancel(request.task_id)
    return AgentCancelResponse(
        task_id=request.task_id,
        cancelled=success,
        message="已发送取消信号" if success else "任务不存在或已完成",
    )


@router.get("/reports")
async def list_agent_reports(limit: int = 50):
    """查询 Agent 执行报告列表"""
    return {"reports": _report_store.list_reports(limit=limit, report_type="agent")}


@router.get("/reports/{report_id}")
async def get_agent_report(report_id: str):
    """获取 Agent 执行报告详情"""
    report = _report_store.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    return report


# ==================== 对话记忆管理 ====================

class MemoryClearRequest(BaseModel):
    """清空会话记忆请求"""
    session_id: str = Field(..., description="要清空记忆的会话 ID")


class MemoryClearResponse(BaseModel):
    """清空会话记忆响应"""
    session_id: str
    cleared: bool
    message: str


@router.post("/memory/clear", response_model=MemoryClearResponse)
async def clear_memory(request: MemoryClearRequest):
    """清空指定会话的持久化对话记忆"""
    try:
        _memory_store.clear(request.session_id)
        # 同步清理内存缓存
        mem = _session_manager.get_or_create(request.session_id)
        mem.clear()
        return MemoryClearResponse(
            session_id=request.session_id,
            cleared=True,
            message="已清空会话记忆",
        )
    except Exception as e:
        logger.warning(f"清空会话记忆失败: {e}")
        return MemoryClearResponse(
            session_id=request.session_id,
            cleared=False,
            message=f"清空失败: {str(e)}",
        )


@router.get("/memory/{session_id}")
async def get_memory(session_id: str, limit: int = 20):
    """获取指定会话的持久化对话历史"""
    history = _memory_store.get_history(session_id, limit=limit)
    return {
        "session_id": session_id,
        "turn_count": len(history),
        "history": history,
    }
