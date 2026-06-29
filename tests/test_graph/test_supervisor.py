"""
Tests for LangGraph Supervisor Graph
=====================================
- supervisor_node: intent classification
- supervisor_router: correct routing
- ReAct loop control
- Forced end on max iterations
"""
from unittest.mock import patch, MagicMock
import pytest

from backend.agent.graph import (
    supervisor_node,
    supervisor_router,
    make_react_reason_node,
    make_react_router,
    after_tools_router,
    execute_tools_node,
    format_output_node,
    _format_forced_end,
    _tools_to_openai_schema,
    MAX_REACT_ITERATIONS,
)
from backend.agent.tools import ALL_TOOLS
from tests.conftest import make_supervisor_state, make_react_state


class TestSupervisorNode:
    """意图分类节点测试"""

    @pytest.mark.parametrize("user_input,expected", [
        ("分析一下为什么登录失败", "log_analysis"),
        ("刚才什么错误", "log_analysis"),
        ("为什么出错了", "rag_query"),
        ("怎么解决登录超时", "rag_query"),
        ("跑一下登录测试", "test_execution"),
        ("执行自动化测试", "test_execution"),
        ("创建一个JIRA缺陷单", "jira_create"),
        ("帮我提个bug", "jira_create"),
    ])
    def test_intent_classification(self, user_input, expected, mock_llm_client):
        """验证 Supervisor 能正确识别用户意图"""
        from backend.agent.graph import supervisor_node

        # mock chat 根据输入返回对应关键词
        def fake_chat(messages, temperature=0.1):
            if "analysis" in user_input or "错误" in user_input:
                return expected
            return expected
        mock_llm_client.chat.side_effect = fake_chat

        state = make_supervisor_state(user_input)
        result = supervisor_node(state)

        assert result["task_type"] == expected, f"Expected {expected}, got {result['task_type']}"
        assert "messages" in result
        assert result["iteration_count"] == 0
        assert result["next_action"] == "llm_reason"

    def test_default_to_rag_query(self, mock_llm_client):
        """无法识别的意图应默认走 rag_query"""
        mock_llm_client.chat.return_value = "some random response without keywords"

        state = make_supervisor_state("Hello world")
        result = supervisor_node(state)

        assert result["task_type"] == "rag_query"


class TestSupervisorRouter:
    """路由节点测试"""

    @pytest.mark.parametrize("task_type,expected_node", [
        ("rag_query", "rag_node"),
        ("log_analysis", "analysis_node"),
        ("test_execution", "test_node"),
        ("jira_create", "jira_node"),
        ("unknown", "rag_node"),
        ("", "rag_node"),
    ])
    def test_route_by_task_type(self, task_type, expected_node):
        """验证路由能按 task_type 正确分发"""
        state = make_supervisor_state("测试", task_type=task_type)
        node = supervisor_router(state)
        assert node == expected_node


class TestReActReasonNode:
    """ReAct 推理节点测试"""

    def test_llm_decides_to_finish(self, mock_llm_client):
        """LLM 没有调用工具时应返回 finish"""
        mock_llm_client.chat_with_tools.return_value = {
            "content": "已完成分析，结果如下...",
            "tool_calls": None,
        }

        reason_fn = make_react_reason_node("test_node", "")
        state = make_react_state(iteration=0)
        result = reason_fn(state)

        assert result["next_action"] == "finish"
        assert "final_response" in result
        assert result["final_response"] != ""

    def test_llm_decides_to_call_tool(self, mock_llm_client):
        """LLM 决定调用工具时应返回 tool_call"""
        mock_llm_client.chat_with_tools.side_effect = None  # override fixture
        mock_llm_client.chat_with_tools.return_value = {
            "content": "让我搜索知识库...",
            "tool_calls": [
                {"id": "call_1", "name": "search_knowledge_base", "args": {"query": "测试"}}
            ],
        }

        reason_fn = make_react_reason_node("test_node", "")
        state = make_react_state(iteration=0)
        result = reason_fn(state)

        assert result["next_action"] == "tool_call"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search_knowledge_base"

    def test_max_iterations_forced_end(self, mock_llm_client):
        """达到最大迭代次数应强制终止"""
        reason_fn = make_react_reason_node("test_node", "")
        state = make_react_state(iteration=MAX_REACT_ITERATIONS)
        result = reason_fn(state)

        assert result["next_action"] == "finish"
        assert len(result.get("final_response", "")) > 0

    def test_llm_error_handling(self, mock_llm_client):
        """LLM 调用失败应返回 error 状态"""
        mock_llm_client.chat_with_tools.side_effect = Exception("API connection failed")

        reason_fn = make_react_reason_node("test_node", "")
        state = make_react_state(iteration=0)
        result = reason_fn(state)

        assert result["next_action"] == "error"
        assert "API connection failed" in result["error"]
        assert "处理请求时发生错误" in result["final_response"]

    def test_extra_context_injected(self, mock_llm_client):
        """extra_context 参数应注入到系统提示词中"""
        mock_llm_client.chat_with_tools.return_value = {
            "content": "ok", "tool_calls": None,
        }

        reason_fn = make_react_reason_node("test_node", extra_context="你是日志分析专家。")
        state = make_react_state()
        reason_fn(state)

        # 验证系统提示包含 extra_context
        call_args = mock_llm_client.chat_with_tools.call_args
        messages = call_args[1]["messages"]
        system_msg = messages[0]["content"]
        assert "你是日志分析专家。" in system_msg


class TestReActRouter:
    """ReAct 循环路由测试"""

    @pytest.mark.parametrize("next_action,expected", [
        ("tool_call", "execute_tools"),
        ("llm_reason", "rag_node"),
        ("finish", "__end__"),
        ("", "__end__"),
    ])
    def test_react_router(self, next_action, expected):
        """验证 ReAct 循环的路由逻辑"""
        router = make_react_router("rag_node")
        state = make_react_state()
        state["next_action"] = next_action
        result = router(state)
        assert result == expected


class TestAfterToolsRouter:
    """工具执行后路由测试"""

    @pytest.mark.parametrize("next_action,origin_node,expected", [
        ("llm_reason", "rag_node", "rag_node"),
        ("llm_reason", "analysis_node", "analysis_node"),
        ("error", "rag_node", "__end__"),
        ("finish", "rag_node", "rag_node"),  # finishes return origin_node, not END
    ])
    def test_after_tools_router(self, next_action, origin_node, expected):
        state = make_react_state(node_name=origin_node)
        state["next_action"] = next_action
        result = after_tools_router(state)
        assert result == expected


class TestExecuteToolsNode:
    """工具执行节点测试"""

    def test_execute_known_tool(self):
        """执行已知工具应返回成功结果"""
        state = make_react_state(tool_calls=[
            {"id": "call_1", "name": "get_system_status", "args": {}},
        ])
        result = execute_tools_node(state)

        assert len(result["tool_results"]) == 1
        assert "tool_call_id" in result["tool_results"][0]
        assert result["tool_results"][0]["tool_call_id"] == "call_1"
        assert result["next_action"] == "llm_reason"

    def test_execute_unknown_tool(self):
        """执行未知工具应返回错误信息"""
        state = make_react_state(tool_calls=[
            {"id": "call_1", "name": "nonexistent_tool", "args": {}},
        ])
        result = execute_tools_node(state)

        assert len(result["tool_results"]) == 1
        assert "未找到" in result["tool_results"][0]["output"]

    def test_multiple_tool_calls(self):
        """多个工具调用应全部执行"""
        state = make_react_state(tool_calls=[
            {"id": "call_1", "name": "get_system_status", "args": {}},
            {"id": "call_2", "name": "list_directory", "args": {"dir_path": "."}},
        ])
        result = execute_tools_node(state)

        assert len(result["tool_results"]) == 2

    def test_empty_tool_calls(self):
        """无工具调用应返回空结果"""
        state = make_react_state(tool_calls=[])
        result = execute_tools_node(state)
        assert len(result["tool_results"]) == 0


class TestFormatOutputNode:
    """输出格式化节点测试"""

    def test_direct_final_response(self):
        """应从 state.final_response 提取输出"""
        state = make_react_state()
        state["final_response"] = "这是最终的输出内容。"
        result = format_output_node(state)
        assert result["final_response"] == "这是最终的输出内容。"

    def test_fallback_to_ai_message(self):
        """无 final_response 时应从最后一条 AI 消息提取"""
        from langchain_core.messages import AIMessage

        state = make_react_state()
        state["messages"] = [
            AIMessage(content="前一条消息"),
            AIMessage(content="最后一条消息"),
        ]
        result = format_output_node(state)
        assert result["final_response"] == "最后一条消息"

    def test_complete_fallback(self):
        """都找不到时应返回默认提示"""
        state = make_react_state()
        result = format_output_node(state)
        assert "处理完成" in result["final_response"]
        assert result["next_action"] == "finish"


class TestAuxFunctions:
    """辅助函数测试"""

    def test_tools_to_openai_schema(self):
        """工具转为 OpenAI schema 应包含所有工具"""
        schemas = _tools_to_openai_schema(ALL_TOOLS)
        assert len(schemas) == len(ALL_TOOLS)
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_tool_descriptions_non_empty(self):
        """每个工具的描述不能为空"""
        assert len(ALL_TOOLS) >= 10  # 至少 10 个工具
        for t in ALL_TOOLS:
            assert t.description, f"Tool '{t.name}' has empty description"

    def test_tools_by_name_index(self):
        """TOOLS_BY_NAME 索引应完整"""
        from backend.agent.tools import TOOLS_BY_NAME
        assert len(TOOLS_BY_NAME) == len(ALL_TOOLS)
        for t in ALL_TOOLS:
            assert t.name in TOOLS_BY_NAME

    def test_format_forced_end_with_results(self):
        """强制结束格式化应包含工具结果"""
        state = make_react_state(tool_results=[
            {"tool_name": "search_knowledge_base", "output": "找到了相关文档", "tool_call_id": "c1"},
        ])
        result = _format_forced_end(state)
        assert "处理摘要" in result
        assert "search_knowledge_base" in result

    def test_format_forced_end_empty(self):
        """无工具结果时返回提示"""
        state = make_react_state()
        result = _format_forced_end(state)
        assert "处理超时" in result
