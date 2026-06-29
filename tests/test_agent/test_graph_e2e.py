"""
Agent Graph 端到端测试
======================
验证 Supervisor 图编译、路由与 ReAct 循环在正常和取消场景下的行为。
依赖 conftest 中已注入的 mock LLMClient。
"""
from __future__ import annotations

import pytest

from backend.agent.graph import build_supervisor_graph, supervisor_router
from backend.agent.state import AgentState
from backend.agent.task_registry import register, cancel, unregister


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
