"""
Agent Graph 端到端测试
======================
验证 Supervisor 图编译、路由与 ReAct 循环在正常和取消场景下的行为。
依赖 conftest 中已注入的 mock LLMClient。
"""
from __future__ import annotations

from backend.agent.graph import build_supervisor_graph, supervisor_router
from backend.agent.task_registry import cancel, register, unregister


class TestSupervisorGraph:
    def test_graph_compiles(self):
        """Supervisor 图应能正常编译"""
        graph = build_supervisor_graph()
        assert graph is not None

    def test_rag_path_end_to_end(self, mock_llm_client):
        """RAG 查询路径应能走完 Supervisor → rag_node → END"""
        from tests.conftest import make_supervisor_state

        graph = build_supervisor_graph()
        initial_state = make_supervisor_state("登录超时怎么办？")

        result = graph.invoke(initial_state)

        assert "final_response" in result
        assert result["final_response"]
        assert result.get("task_type") == "rag_query"

    def test_supervisor_router_routes_by_task_type(self):
        """supervisor_router 根据 task_type 返回正确节点"""
        assert supervisor_router({"task_type": "rag_query"}) == "rag_node"
        assert supervisor_router({"task_type": "log_analysis"}) == "analysis_node"
        assert supervisor_router({"task_type": "test_execution"}) == "test_node"
        assert supervisor_router({"task_type": "jira_create"}) == "jira_node"
        assert supervisor_router({"task_type": "unknown"}) == "rag_node"


class TestTaskCancellation:
    def test_cancel_signal_ends_react(self, mock_llm_client):
        """任务取消后，ReAct 节点应提前返回 finish"""
        from tests.conftest import make_supervisor_state

        task_id = "cancel-test-001"
        register(task_id)

        graph = build_supervisor_graph()
        initial_state = make_supervisor_state("执行登录测试")
        initial_state["task_id"] = task_id
        initial_state["task_type"] = "test_execution"

        cancel(task_id)
        result = graph.invoke(initial_state)

        assert "final_response" in result
        assert "取消" in result["final_response"]

        unregister(task_id)


class TestToolCallsNoReplay:
    """回归：tool_calls 必须是替换语义 — 每轮只执行本轮工具调用，禁止重放历史。

    事故背景（2026-08-23 app.log 8c57f5f1）：tool_calls 曾为 operator.add
    累积语义，第 2 轮 ReAct 重放第 1 轮的 explore_website，导致百度页面
    被重复打开两次。修复：state.py 中 tool_calls 改为普通替换语义。
    """

    def test_each_round_executes_only_current_calls(self, mock_llm_client):
        """两轮 ReAct 各决定 1 个工具 → 每个工具各执行 1 次（共 2 次），不重放"""
        from unittest.mock import patch

        from tests.conftest import make_supervisor_state

        # Supervisor 分类 → test_execution
        mock_llm_client.chat.return_value = "test_execution"

        # ReAct 轮次序列：轮1 工具A → 轮2 工具B → 轮3 最终回答
        mock_llm_client.chat_with_tools.side_effect = [
            {"content": "", "tool_calls": [{"name": "probe_tool_a", "args": {}, "id": "a1"}]},
            {"content": "", "tool_calls": [{"name": "probe_tool_b", "args": {}, "id": "b1"}]},
            {"content": "测试完成", "tool_calls": None},
        ]

        executed = []

        def make_probe(tag):
            def invoke(args):
                executed.append(tag)
                return f"工具{tag}结果"
            return type(f"Probe{tag.upper()}", (), {"invoke": staticmethod(invoke)})()

        import backend.agent.tools as tools_module

        with patch.dict(
            tools_module.TOOLS_BY_NAME,
            {"probe_tool_a": make_probe("a"), "probe_tool_b": make_probe("b")},
        ):
            graph = build_supervisor_graph()
            result = graph.invoke(make_supervisor_state("执行测试"))

        assert result["final_response"] == "测试完成"
        # 修复前（累积语义）为 ["a", "a", "b"]——第 2 轮重放工具 a
        assert executed == ["a", "b"]
