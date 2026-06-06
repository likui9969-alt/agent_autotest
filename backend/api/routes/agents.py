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
from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.agent.graph import get_supervisor_graph
from backend.agent.state import AgentState

logger = logging.getLogger("ai_rd_agent")
router = APIRouter(tags=["Agent执行"])


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


class AgentExecuteResponse(BaseModel):
    """Agent 执行响应"""
    task_id: str
    task_type: str
    final_response: str
    tool_calls_made: int
    iterations: int
    execution_time_ms: float
    error: str = ""


# ==================== 核心执行逻辑 ====================

def _execute_agent(task: str, max_iterations: int) -> dict:
    """执行 Agent 任务的核心逻辑

    Args:
        task: 用户任务描述
        max_iterations: ReAct 最大迭代次数

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
        "context": {},
        "final_response": "",
        "next_action": "",
        "error": "",
    }

    # 执行图
    result = graph.invoke(initial_state, {"recursion_limit": 50})

    # 提取结果
    tool_calls_made = len(result.get("tool_results", []))
    iterations = result.get("iteration_count", 0)

    return {
        "task_type": result.get("task_type", "unknown"),
        "final_response": result.get("final_response", ""),
        "tool_calls_made": tool_calls_made,
        "iterations": iterations,
        "error": result.get("error", ""),
        "messages": result.get("messages", []),
        "tool_results": result.get("tool_results", []),
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

    logger.info(f"[Agent API {task_id}] 收到任务: {request.task[:100]}...")

    try:
        # 在线程池中执行（避免阻塞事件循环）
        result = await asyncio.to_thread(
            _execute_agent,
            request.task,
            request.max_iterations,
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
                "context": {},
                "final_response": "",
                "next_action": "",
                "error": "",
            }

            # 使用 stream() 逐节点推送
            tool_count = 0
            iter_count = 0
            final_response = ""

            for event in graph.stream(initial_state, {"recursion_limit": 50}):
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

                    yield _sse_event("node_end", {
                        "node": node_name,
                        "next_action": state_update.get("next_action", ""),
                    })

            # 发送最终结果
            yield _sse_event("final", {
                "task_type": "completed",
                "response": final_response,
                "tool_calls": tool_count,
                "iterations": iter_count,
                "time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            })

        except Exception as e:
            logger.error(f"[Agent SSE {task_id}] 错误: {e}", exc_info=True)
            yield _sse_event("error", {"message": str(e)})

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


def _execute_agent_streaming(task: str, max_iterations: int, event_callback=None) -> dict:
    """带回调的执行函数（支持流式推送中间状态）"""
    graph = get_supervisor_graph()

    initial_state: AgentState = {
        "messages": [],
        "task_type": "unknown",
        "user_input": task,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "tool_calls": [],
        "tool_results": [],
        "context": {},
        "final_response": "",
        "next_action": "",
        "error": "",
    }

    result = graph.invoke(initial_state, {"recursion_limit": 50})

    return {
        "task_type": result.get("task_type", "unknown"),
        "final_response": result.get("final_response", ""),
        "tool_calls_made": len(result.get("tool_results", [])),
        "iterations": result.get("iteration_count", 0),
        "error": result.get("error", ""),
    }


def _sse_event(event_type: str, data: dict) -> str:
    """构建 SSE 格式的事件字符串"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
