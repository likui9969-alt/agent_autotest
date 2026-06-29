"""Agent 执行页。"""
from __future__ import annotations

import json
import logging as _logging

import requests
import streamlit as st

from utils.api import api_post, api_post_raw, get_api_host


def render() -> None:
    API_BASE = get_api_host()

    st.title("🤖 Agent 执行")
    st.markdown("""
    **LangGraph Supervisor + ReAct 推理循环**

    输入任务描述，AI Agent 自动：
    1. 分析意图 → 2. 选择工具 → 3. 执行工具 → 4. 观察结果 → 5. 继续推理 → 6. 生成最终回答

    可用工具：`search_knowledge_base` | `parse_log_content` | `execute_test_scenario` | `run_real_test_scenario` | `create_jira_issue_tool` | `get_runtime_logs` | `get_system_status` | `read_code_file` | `list_directory` | `run_shell_command` | `check_api_health` | `get_recent_test_logs` | `explore_website` | `run_custom_test`
    """)

    # ---- 会话管理 ----
    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id = _new_session_id()

    session_id = st.session_state.agent_session_id

    with st.expander("💬 对话会话管理", expanded=False):
        st.caption("同一 session_id 的多轮对话会被持久化，Agent 能自动引用历史上下文。")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.text_input("当前会话 ID", value=session_id, key="agent_session_id_input", disabled=True)
        with c2:
            if st.button("🆕 新建会话", use_container_width=True, help="生成新的 session_id，开始全新对话"):
                st.session_state.agent_session_id = _new_session_id()
                st.session_state.agent_events = []
                st.session_state.pop("agent_result", None)
                st.rerun()
        with c3:
            if st.button("🗑️ 清空历史", use_container_width=True, help="保留当前会话 ID，仅清空已保存的历史记录"):
                _clear_session_memory(session_id)
                st.session_state.agent_events = []
                st.session_state.pop("agent_result", None)
                st.success("已清空当前会话历史")
                st.rerun()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("任务描述")

        # 初始化任务文本，避免 session state 异常导致读取到页面其他内容
        if "agent_task" not in st.session_state:
            st.session_state.agent_task = ""

        # 快捷示例 — 一键填充常见任务
        st.caption("💡 点击示例快速填充：")
        ex_cols = st.columns(3)
        examples = {
            "🚨 刚才什么错误": {"text": "刚才后端运行出了什么错误？请读取运行日志并分析原因和解决方案。", "help": "读取最近 ERROR 级别运行日志并分析"},
            "🔍 系统状态检查": {"text": "检查一下系统各组件状态是否正常，LLM、向量库、Chrome 是否可用。", "help": "调用系统状态诊断工具"},
            "🧪 跑测试并分析": {"text": "执行登录和搜索场景的自动化测试，如果失败请分析原因并给出修复建议。", "help": "执行内置 Demo 测试并分析结果"},
        }
        for c, (label, info) in zip(ex_cols, examples.items()):
            if c.button(label, key=f"ex_{label}", help=info["help"], use_container_width=True):
                st.session_state.agent_task_input = info["text"]
                st.rerun()

        st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)

        task = st.text_area(
            "用自然语言描述你想让 Agent 完成的任务",
            value=st.session_state.agent_task,
            height=120,
            placeholder="例如：\n"
                       "• 刚才什么错误？读取运行日志分析一下\n"
                       "• 分析这个错误：TimeoutException: Page load timeout after 30s\n"
                       "• 执行登录测试，分析失败原因，确认是 bug 就创建 JIRA 缺陷单",
            key="agent_task_input",
        )
        # 同步回 session_state，让示例按钮可以正确填充
        st.session_state.agent_task = task

        # 清理任务文本：去除首尾空白、常见误输入的 UI 标签
        task_clean = task.strip()
        ui_noise = [
            "输入任务描述，点击执行按钮启动 Agent",
            "什么是 ReAct 循环？",
            "Agent 不会一次性给出答案",
            "ReAct 最大迭代次数",
            "任务类型（可选）",
            "🤖 自动识别",
            "执行结果",
        ]
        for noise in ui_noise:
            if noise in task_clean:
                task_clean = task_clean.split(noise)[0].strip()

        max_iters = st.slider("ReAct 最大迭代次数", 1, 10, 5, key="agent_iters",
                              help="Agent 最多进行多少轮 思考→工具→观察 循环")

        task_type = st.selectbox(
            "任务类型（可选）",
            ["auto", "rag_query", "log_analysis", "test_execution", "jira_create"],
            format_func=lambda x: {
                "auto": "🤖 自动识别",
                "rag_query": "📚 知识库问答",
                "log_analysis": "📊 日志分析",
                "test_execution": "🧪 测试执行",
                "jira_create": "🎫 JIRA 创建"
            }[x],
            key="agent_task_type",
        )

        use_stream = st.checkbox("流式输出（SSE）", value=True, key="agent_stream",
                                 help="实时展示 Agent 的推理过程和工具调用")

        # 如果任务为空，给出明确的视觉提示
        if not task_clean:
            st.warning("⚠️ 请先输入任务描述，或点击上方示例按钮")

        exec_col, stop_col = st.columns([3, 1])
        with exec_col:
            run_clicked = st.button("🚀 执行 Agent 任务", type="primary", disabled=not task_clean, key="agent_exec_btn", use_container_width=True)
        with stop_col:
            def _cancel_agent():
                task_id = st.session_state.get("agent_task_id", "")
                if task_id:
                    try:
                        requests.post(f"{API_BASE}/api/v1/agent/cancel", json={"task_id": task_id}, timeout=5)
                    except Exception:
                        pass
                st.session_state.pop("agent_task_id", None)
                st.session_state.pop("agent_result", None)
                st.session_state["agent_events"] = []
            st.button("⏹ 停止", type="secondary", key="agent_stop_btn", use_container_width=True, on_click=_cancel_agent)

        if run_clicked:
            # 清空上一次的结果
            st.session_state.agent_events = []
            st.session_state.pop("agent_result", None)
            st.session_state["agent_task_id"] = ""

            with st.spinner("Agent 推理中..."):
                if use_stream:
                    # SSE 流式模式
                    stream_result = None
                    current_event_type = None
                    try:
                        resp = api_post_raw(
                            "api/v1/agent/execute/stream",
                            json={
                                "task": task_clean,
                                "max_iterations": max_iters,
                                "task_type": task_type,
                                "session_id": session_id,
                            },
                            stream=True,
                            timeout=300,
                        )
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line:
                                current_event_type = None
                                continue
                            if line.startswith("event:"):
                                current_event_type = line[6:].strip()
                            elif line.startswith("data:"):
                                data_str = line[5:].strip()
                                if data_str:
                                    try:
                                        event_data = json.loads(data_str)
                                        if current_event_type == "start" and event_data.get("task_id"):
                                            st.session_state["agent_task_id"] = event_data["task_id"]
                                        st.session_state.agent_events.append({
                                            "event": current_event_type,
                                            "data": event_data,
                                        })
                                        if current_event_type == "final":
                                            stream_result = {
                                                "task_type": event_data.get("task_type", ""),
                                                "final_response": event_data.get("response", ""),
                                                "tool_calls_made": event_data.get("tool_calls", 0),
                                                "iterations": event_data.get("iterations", 0),
                                                "execution_time_ms": event_data.get("time_ms", 0),
                                                "task_id": st.session_state.get("agent_task_id", ""),
                                            }
                                    except Exception as e:
                                        # SSE 事件解析失败时跳过该事件，不影响后续处理
                                        _logging.getLogger("frontend").debug(f"SSE 事件解析跳过: {e}")
                    except Exception as e:
                        st.error(f"SSE 连接失败: {str(e)}")

                    if stream_result:
                        st.session_state.agent_result = stream_result
                        st.rerun()
                    elif st.session_state.agent_events:
                        # 流式有事件但无 final，仍展示已收集的推理过程
                        st.warning("流式未返回 final 事件，但已记录推理过程（见右侧）")
                    else:
                        st.info("流式未返回结果，切换到普通模式...")

                # 非流式模式（流式未启用或流式未成功时执行）
                if not use_stream or "agent_result" not in st.session_state:
                    success, data = api_post(
                        "api/v1/agent/execute",
                        json={
                            "task": task_clean,
                            "max_iterations": max_iters,
                            "task_type": task_type,
                            "session_id": session_id,
                        },
                        timeout=300,
                    )
                    if success:
                        st.session_state["agent_task_id"] = data.get("task_id", "")
                        st.session_state.agent_result = data
                        st.rerun()
                    else:
                        st.error(f"执行失败: {data}")

    with col2:
        st.subheader("执行结果")
        has_result = "agent_result" in st.session_state and st.session_state.agent_result
        has_events = bool(st.session_state.get("agent_events"))

        if has_result:
            result = st.session_state.agent_result

            # 统计卡片
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("任务类型", result.get("task_type", "N/A"))
            with m2:
                st.metric("工具调用", result.get("tool_calls_made", 0))
            with m3:
                st.metric("推理轮次", result.get("iterations", 0))
            with m4:
                st.metric("Token", result.get("total_tokens", 0))
            with m5:
                st.metric("耗时", f"{result.get('execution_time_ms', 0):.0f}ms")

            # 错误信息
            if result.get("error"):
                st.error(f"执行错误: {result['error']}")

            # ---- 推理过程时间线（来自 SSE 事件） ----
            if has_events:
                with st.expander("🧠 推理过程时间线", expanded=True):
                    _render_agent_timeline(st.session_state.agent_events)

            # 最终回答
            st.markdown("### 💡 Agent 回答")
            final_response = result.get("final_response", "")
            if final_response:
                st.markdown(final_response)
            else:
                st.warning("Agent 未返回有效回答")

            # 任务信息
            with st.expander("📋 执行详情"):
                st.json({
                    "task_id": result.get("task_id", ""),
                    "task_type": result.get("task_type", ""),
                    "tool_calls_made": result.get("tool_calls_made", 0),
                    "iterations": result.get("iterations", 0),
                    "time_ms": result.get("execution_time_ms", 0),
                })
        elif has_events:
            # 仅有事件无最终结果（流式中断场景）
            st.info(f"已记录 {len(st.session_state.agent_events)} 个推理事件（未获得最终回答）")
            with st.expander("🧠 推理过程时间线", expanded=True):
                _render_agent_timeline(st.session_state.agent_events)
        else:
            st.info("👈 输入任务描述，点击执行按钮启动 Agent")
            with st.expander("💡 什么是 ReAct 循环？"):
                st.markdown(
                    "Agent 不会一次性给出答案，而是：\n\n"
                    "1. 🧠 **Think** — 分析问题，决定要用什么工具\n"
                    "2. 🔧 **Act** — 调用工具（搜知识库/跑测试/读运行日志/查系统状态/创缺陷单）\n"
                    "3. 👁 **Observe** — 查看工具返回的结果\n"
                    "4. 🔄 重复以上步骤，直到获得足够信息给出最终答案"
                )


def _new_session_id() -> str:
    """生成新的会话 ID（短 UUID）。"""
    import uuid
    return str(uuid.uuid4())[:8]


def _clear_session_memory(session_id: str) -> None:
    """调用后端接口清空当前会话的持久化记忆。"""
    try:
        requests.post(
            f"{get_api_host()}/api/v1/agent/memory/clear",
            json={"session_id": session_id},
            timeout=5,
        )
    except Exception:
        pass


def _render_agent_timeline(events: list[dict]):
    """将 SSE 事件列表渲染为可视化的推理时间线

    事件类型：
    - start       : 任务开始
    - node_start  : 进入图节点（supervisor / rag_node / execute_tools ...）
    - tool_call   : Agent 调用工具
    - node_end    : 节点执行完毕
    - final       : 最终回答
    - error       : 错误
    - done        : 流结束
    """
    if not events:
        st.caption("（无推理事件）")
        return

    node_icons = {
        "supervisor": "🎯",
        "rag_node": "📚",
        "analysis_node": "📊",
        "test_node": "🧪",
        "jira_node": "🎫",
        "execute_tools": "🔧",
        "format_output": "📝",
    }

    step_idx = 0
    for ev in events:
        etype = ev.get("event", "")
        data = ev.get("data", {}) or {}

        if etype == "start":
            st.markdown(f"**▶ 任务开始** — `{data.get('task', '')[:80]}`")
            st.caption(f"task_id: {data.get('task_id', '')} | 最大迭代: {data.get('max_iterations', '')}")

        elif etype == "node_start":
            step_idx += 1
            node = data.get("node", "")
            icon = node_icons.get(node, "⚙️")
            it = data.get("iteration", 0)
            label = {
                "supervisor": "意图分类",
                "rag_node": "RAG 问答推理",
                "analysis_node": "日志分析推理",
                "test_node": "测试执行推理",
                "jira_node": "JIRA 创建推理",
                "execute_tools": "执行工具",
                "format_output": "格式化输出",
            }.get(node, node)
            st.markdown(f"**{icon} 步骤 {step_idx}：{label}**" + (f" · 第 {it} 轮" if it else ""))

        elif etype == "tool_call":
            tool = data.get("tool", "unknown")
            args = data.get("args", {})
            args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else "(无参数)"
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ 🔧 调用工具 **`{tool}`**({args_str})")

        elif etype == "node_end":
            nxt = data.get("next_action", "")
            if nxt:
                st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;→ 下一步: `{nxt}`")

        elif etype == "final":
            st.markdown(
                f"**✅ 完成** — 工具调用 {data.get('tool_calls', 0)} 次 | "
                f"迭代 {data.get('iterations', 0)} 轮 | "
                f"耗时 {data.get('time_ms', 0):.0f}ms"
            )

        elif etype == "error":
            st.error(f"❌ 错误: {data.get('message', '')}")

        elif etype == "done":
            st.caption("— 流结束 —")


if __name__ == "__main__":
    render()
