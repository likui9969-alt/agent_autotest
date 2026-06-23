"""
LangGraph Supervisor — 多 Agent 协作状态机

架构:
    START → supervisor (意图分类)
              ├──→ rag_node (ReAct 循环)
              ├──→ analysis_node (ReAct 循环)
              ├──→ test_node (ReAct 循环)
              └──→ jira_node (ReAct 循环)
                        ↓
                    END

每个专业节点内部运行 ReAct 循环:
    llm_reason → has_tool_call?
                  ├── YES → execute_tools → llm_reason
                  └── NO  → END
"""
import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from backend.agent.state import AgentState
from backend.agent.tools import ALL_TOOLS
from backend.llm.prompts import get_template

# LangChain → OpenAI 角色映射
ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}

logger = logging.getLogger("ai_rd_agent")

# ReAct 循环最大迭代次数
MAX_REACT_ITERATIONS = 5

# ==================== 系统提示词 ====================

SUPERVISOR_SYSTEM_PROMPT = """你是一个 AI 研发效能智能体的调度中心（Supervisor）。

你的职责是理解用户意图，将任务分发给对应的专业 Agent。

任务分类规则：
- 用户问"刚才什么错误/发生了什么/最近什么异常/为什么失败" + 排查问题 → log_analysis
- 用户上传日志/要求分析日志/排查故障 → log_analysis
- 用户问"怎么解决/如何处理/为什么出错" + 技术问题 → rag_query
- 用户要求执行测试/跑自动化/验证功能 → test_execution
- 用户要求创建缺陷单/提 bug/创建 JIRA → jira_create
- 用户问"系统状态/环境是否正常/Chrome 是否可用" → rag_query
- 用户的问题包含多个步骤 → 按优先级依次处理

请只回复任务类型（rag_query / log_analysis / test_execution / jira_create），不要回复其他内容。"""


REACT_SYSTEM_PROMPT = """你是一个智能 Agent，拥有以下工具可以调用：

{tool_descriptions}

处理用户请求时，请遵循 ReAct 模式：
1. Thought（思考）：分析当前情况，决定下一步
2. Action（行动）：调用合适的工具
3. Observation（观察）：分析工具返回结果
4. 重复以上步骤直到能给出最终答案

重要规则：
- 每次只调用一个工具
- 最多进行 {max_iterations} 轮工具调用
- 获得足够信息后，用中文给出完整、结构化的最终回答
- 如果工具返回错误，尝试其他方法而非放弃

输出格式：
- 需要调用工具时：说明为什么调用 + 调用哪个工具
- 不需要工具时：直接给出完整回答"""


# ==================== Supervisor 节点 ====================

def supervisor_node(state: AgentState) -> dict:
    """意图分类节点 — 分析用户输入，决定路由到哪个专业 Agent"""
    user_input = state.get("user_input", "")

    logger.info(f"[Supervisor] 分类用户意图: {user_input[:100]}...")

    from backend.api.deps import get_llm_client
    llm = get_llm_client()
    response = llm.chat([
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ], temperature=0.1).strip().lower()

    # 解析分类结果
    task_map = {
        "rag_query": "rag_query",
        "log_analysis": "log_analysis",
        "test_execution": "test_execution",
        "jira_create": "jira_create",
    }
    task_type = "rag_query"  # 默认走 RAG
    for key, value in task_map.items():
        if key in response:
            task_type = value
            break

    logger.info(f"[Supervisor] 分类结果: {task_type}")

    return {
        "task_type": task_type,
        "messages": [HumanMessage(content=user_input)],
        "iteration_count": 0,
        "max_iterations": MAX_REACT_ITERATIONS,
        "context": {},
        "next_action": "llm_reason",
        "error": "",
    }


def supervisor_router(state: AgentState) -> Literal["rag_node", "analysis_node", "test_node", "jira_node"]:
    """根据意图分类路由到对应专业节点"""
    task_type = state.get("task_type", "rag_query")
    route_map = {
        "rag_query": "rag_node",
        "log_analysis": "analysis_node",
        "test_execution": "test_node",
        "jira_create": "jira_node",
    }
    return route_map.get(task_type, "rag_node")


# ==================== ReAct 推理节点（通用） ====================

def make_react_reason_node(node_name: str, extra_context: str = "", node_key: str = ""):
    """工厂函数 — 创建带 ReAct 循环的推理节点

    Args:
        node_name: 节点名称（用于日志）
        extra_context: 额外的系统提示（注入到 ReAct prompt 中）
        node_key: 图节点名（用于路由回到自身），默认与 node_name 相同

    Returns:
        推理函数，可注册到 LangGraph
    """
    _node_key = node_key or node_name

    def react_reason(state: AgentState) -> dict:
        """ReAct 推理步骤 — 决定下一步：调用工具 or 结束"""
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", MAX_REACT_ITERATIONS)
        messages = list(state.get("messages", []))

        logger.info(f"[{node_name}] ReAct 推理 第 {iteration + 1}/{max_iter} 轮")

        # 超过最大迭代次数，强制终止
        if iteration >= max_iter:
            logger.warning(f"[{node_name}] 达到最大迭代次数 {max_iter}，强制终止")
            return {
                "next_action": "finish",
                "final_response": _format_forced_end(state),
            }

        # 构建工具描述
        tool_descriptions = "\n".join(
            f"- **{t.name}**: {t.description}" for t in ALL_TOOLS
        )

        system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            max_iterations=max_iter,
        )
        if extra_context:
            system_prompt += f"\n\n额外上下文：\n{extra_context}"

        # 构建消息列表
        full_messages = [SystemMessage(content=system_prompt)]
        # 加入已执行的工具结果作为上下文
        for msg in messages[-10:]:  # 只保留最近 10 条，防止 token 溢出
            full_messages.append(msg)

        # 调用 LLM（带工具绑定）
        from backend.api.deps import get_llm_client
        llm = get_llm_client()
        try:
            tools_schema = _tools_to_openai_schema(ALL_TOOLS)
            result = llm.chat_with_tools(
                messages=[{"role": ROLE_MAP.get(m.type, "user"),
                           "content": m.content} for m in full_messages],
                tools=tools_schema,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=4096,
            )

            # 检查是否有工具调用
            if result["tool_calls"]:
                tool_call_list = result["tool_calls"]

                logger.info(f"[{node_name}] LLM 决定调用工具: {[t['name'] for t in tool_call_list]}")

                return {
                    "messages": [AIMessage(
                        content=result["content"],
                        tool_calls=tool_call_list,
                    )],
                    "tool_calls": tool_call_list,
                    "iteration_count": iteration + 1,
                    "next_action": "tool_call",
                    "origin_node": _node_key,
                }
            else:
                # 没有工具调用，LLM 给出了最终回答
                logger.info(f"[{node_name}] LLM 输出最终回答")
                return {
                    "messages": [AIMessage(content=result["content"])],
                    "final_response": result["content"],
                    "iteration_count": iteration + 1,
                    "next_action": "finish",
                }

        except Exception as e:
            logger.error(f"[{node_name}] LLM 调用失败: {e}", exc_info=True)
            return {
                "next_action": "error",
                "error": str(e),
                "final_response": f"处理请求时发生错误: {str(e)}",
            }

    return react_reason


# ==================== 工具执行节点 ====================

def execute_tools_node(state: AgentState) -> dict:
    """执行 Agent 请求的工具调用，返回观察结果"""
    tool_calls = state.get("tool_calls", [])
    results = []

    logger.info(f"[ToolExecutor] 执行 {len(tool_calls)} 个工具调用")

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "")

        # 查找工具
        from backend.agent.tools import TOOLS_BY_NAME
        tool_fn = TOOLS_BY_NAME.get(tool_name)

        if tool_fn is None:
            output = f"错误: 未找到工具 '{tool_name}'"
            logger.warning(f"[ToolExecutor] 未知工具: {tool_name}")
        else:
            try:
                # 执行工具
                logger.info(f"[ToolExecutor] 调用 {tool_name}({list(tool_args.keys())})")
                result = tool_fn.invoke(tool_args)
                output = str(result)
                logger.info(f"[ToolExecutor] {tool_name} 完成 ({len(output)} 字符)")
            except Exception as e:
                output = f"工具执行失败: {str(e)}"
                logger.error(f"[ToolExecutor] {tool_name} 失败: {e}")

        results.append({
            "tool_call_id": tool_id,
            "tool_name": tool_name,
            "output": output[:4000],  # 限制长度
        })

    # 构建观察消息
    observation_text = "\n\n".join(
        f"[工具: {r['tool_name']}]\n{r['output']}" for r in results
    )

    return {
        "tool_results": results,
        "messages": [HumanMessage(content=f"工具执行结果:\n{observation_text}")],
        "next_action": "llm_reason",
    }


# ==================== 路由判断 ====================

def make_react_router(node_name: str):
    """工厂函数 — 创建 ReAct 循环路由，直接返回目标节点名，消除字符串映射断裂风险"""
    def react_router(state: AgentState) -> str:
        next_action = state.get("next_action", "finish")

        if next_action == "tool_call":
            return "execute_tools"
        elif next_action == "llm_reason":
            return node_name  # 直接返回节点名，不经过抽象字符串映射
        return END

    return react_router


def after_tools_router(state: AgentState) -> Literal["rag_node", "analysis_node", "test_node", "jira_node", END]:
    """工具执行后的路由 — 根据 origin_node 回到发起工具调用的节点"""
    next_action = state.get("next_action", "llm_reason")
    if next_action == "error":
        return END
    # 回到发起工具调用的节点，继续 ReAct 循环
    origin = state.get("origin_node", "analysis_node")
    return origin  # type: ignore[return-value]


# ==================== 结束格式化 ====================

def format_output_node(state: AgentState) -> dict:
    """格式化最终输出"""
    final = state.get("final_response", "")
    if not final:
        # 尝试从最后一条 AI 消息中提取
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                final = msg.content
                break

    if not final:
        final = "处理完成，但未能生成回答。请重试。"

    return {
        "final_response": final,
        "next_action": "finish",
    }


# ==================== 图构建 ====================

def build_supervisor_graph() -> StateGraph:
    """构建完整的多 Agent 协作图

    返回编译后的 StateGraph，可直接调用 invoke() / stream()
    """
    # 创建图
    workflow = StateGraph(AgentState)

    # ---- 注册节点 ----
    # Supervisor
    workflow.add_node("supervisor", supervisor_node)

    # 4 个专业 Agent（每个内部有独立的 ReAct 循环）
    workflow.add_node("rag_node", make_react_reason_node("RAG",
        "你是 RAG 问答专家。用户遇到问题时，先用 search_knowledge_base 搜索知识库，"
        "然后基于检索结果给出分析、原因和解决方案。引用知识库中的具体案例。"
        "如果用户问'刚才什么错误/发生了什么'，先用 get_runtime_logs(level='ERROR') 读取运行日志，"
        "再用 parse_log_content 解析异常，最后搜索知识库给出解决方案。",
        node_key="rag_node"))

    workflow.add_node("analysis_node", make_react_reason_node("Analysis",
        "你是日志分析专家。如果用户没有提供日志内容但问'刚才什么错误/最近什么异常'，"
        "先用 get_runtime_logs(level='ERROR') 读取后端运行日志；"
        "拿到日志后用 parse_log_content 解析异常，"
        "然后用 search_knowledge_base 搜索历史相似案例，"
        "最后给出完整的故障分析报告（原因+案例+修复建议）。",
        node_key="analysis_node"))

    workflow.add_node("test_node", make_react_reason_node("Test",
        "你是自动化测试专家。根据需要执行测试场景。"
        "如果不确定环境是否就绪，可先用 get_system_status 检查 Chrome/chromedriver 是否可用。"
        "然后用 execute_test_scenario 执行测试，分析结果，"
        "如果失败则进一步分析原因并给出修复建议。",
        node_key="test_node"))

    workflow.add_node("jira_node", make_react_reason_node("JIRA",
        "你是缺陷管理专家。根据分析结果，"
        "用 create_jira_issue_tool 创建 JIRA 缺陷单，"
        "标题简洁明了，描述包含复现步骤和 AI 分析结果。",
        node_key="jira_node"))

    # 工具执行节点（所有 Agent 共享）
    workflow.add_node("execute_tools", execute_tools_node)

    # 格式化输出节点
    workflow.add_node("format_output", format_output_node)

    # ---- 设置入口 ----
    workflow.set_entry_point("supervisor")

    # ---- Supervisor 路由到专业 Agent ----
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "rag_node": "rag_node",
            "analysis_node": "analysis_node",
            "test_node": "test_node",
            "jira_node": "jira_node",
        }
    )

    # ---- 每个专业 Agent 的 ReAct 循环 — 每个节点绑定自己的路由实例 ----
    for node_name in ["rag_node", "analysis_node", "test_node", "jira_node"]:
        workflow.add_conditional_edges(
            node_name,
            make_react_router(node_name),  # 工厂函数，直接返回节点名
            {
                node_name: node_name,       # 继续思考（循环回自身）
                "execute_tools": "execute_tools",  # 调用工具
                END: "format_output",       # 结束
            }
        )

    # ---- 工具执行后回到发起工具调用的节点（继续 ReAct 循环） ----
    workflow.add_conditional_edges(
        "execute_tools",
        after_tools_router,
        {
            "rag_node": "rag_node",
            "analysis_node": "analysis_node",
            "test_node": "test_node",
            "jira_node": "jira_node",
            END: "format_output",
        }
    )

    # 格式化输出后结束
    workflow.add_edge("format_output", END)

    # 编译图
    compiled = workflow.compile()
    logger.info("Supervisor Graph 已编译完成")
    return compiled


# ==================== 辅助函数 ====================

def _tools_to_openai_schema(tools: list) -> list[dict]:
    """将 LangChain 工具转换为 OpenAI function calling 格式"""
    schemas = []
    for t in tools:
        # 从 LangChain tool 提取参数 schema
        if hasattr(t, "args_schema") and t.args_schema:
            params = t.args_schema.model_json_schema()
        else:
            params = {"type": "object", "properties": {}}

        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description[:1024],  # OpenAI 限制
                "parameters": params,
            }
        })
    return schemas


def _format_forced_end(state: AgentState) -> str:
    """当达到最大迭代次数时，整理已有信息输出"""
    tool_results = state.get("tool_results", [])
    if not tool_results:
        return "处理超时，但未获得足够信息。请尝试提供更具体的问题。"

    parts = ["## 处理摘要（已达到最大推理轮次）\n"]
    for r in tool_results:
        parts.append(f"### {r['tool_name']}\n{r['output'][:500]}\n")

    return "\n".join(parts)


# ==================== 全局图实例（单例） ====================

_graph_instance = None


def get_supervisor_graph() -> StateGraph:
    """获取 Supervisor 图的全局单例"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_supervisor_graph()
    return _graph_instance
