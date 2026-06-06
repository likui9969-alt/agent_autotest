"""
AgentState — LangGraph 全局状态定义

状态在各节点间流转，所有字段通过 operator.add 实现增量追加
而非覆盖，保证并行节点安全合并。
"""
import operator
from typing import TypedDict, Annotated, Any
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """LangGraph 多 Agent 协作的全局共享状态

    每个节点从 state 读取所需字段，处理后返回增量更新。
    Annotated[list, operator.add] 确保多节点并行时结果合并而非覆盖。
    """

    # ---- 对话历史（增量追加） ----
    messages: Annotated[list[BaseMessage], operator.add]

    # ---- 任务分类 ----
    task_type: str
    # 取值: "rag_query" | "log_analysis" | "test_execution" | "jira_create" | "unknown"

    # ---- 用户原始输入 ----
    user_input: str

    # ---- ReAct 循环控制 ----
    iteration_count: int
    # 当前迭代轮次，达到 MAX_ITERATIONS 时强制终止
    max_iterations: int
    # 最大允许的 think→act→observe 循环次数，默认 5

    # ---- 工具调用追踪 ----
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    # 本轮待执行的工具调用 [{name, args, id}, ...]
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    # 工具执行结果 [{tool_name, output, error}, ...]

    # ---- 上下文/中间结果 ----
    context: dict[str, Any]
    # 各节点共享的上下文数据:
    #   - retrieved_docs: RAG 检索到的文档
    #   - exceptions_found: 日志分析发现的异常
    #   - test_results: 测试执行结果
    #   - analysis_result: 故障分析结果

    # ---- 最终输出 ----
    final_response: str
    # 最终返回给用户的文本

    # ---- 路由控制 ----
    next_action: str
    # 取值: "tool_call" | "llm_reason" | "finish" | "error"

    # ---- 错误信息 ----
    error: str
    # 异常发生时记录错误描述
