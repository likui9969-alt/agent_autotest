"""
API 集成测试
============
覆盖 /api/v1/testing 和 /api/v1/knowledge 的核心端点，
通过依赖注入覆盖避免真实调用 Selenium / Chroma / LLM。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.main import app
from backend.api.deps import get_rag_pipeline, get_test_executor
from backend.models.testing import TestCaseResult, TestReport, TestStatus


# ---- Fake 依赖 ----

class FakeTestExecutor:
    """不启动浏览器的测试执行器"""

    def __init__(self, *args, **kwargs):
        pass

    def run_tests(self, request, cancel_event=None):
        return TestReport(
            report_id="report-fake",
            base_url=request.base_url,
            total_scenarios=1,
            passed_count=1,
            failed_count=0,
            pass_rate=1.0,
            results=[
                TestCaseResult(
                    scenario="login",
                    status=TestStatus.PASSED,
                    duration_ms=120.0,
                )
            ],
        )


class FakeRAGPipeline:
    """不访问真实 Chroma 的 RAG 管线"""

    def stats(self):
        return {
            "collection_name": "knowledge_base",
            "total_chunks": 10,
            "persist_directory": "/tmp/fake_chroma",
        }

    def ingest_directory_incremental(self, dir_path: str):
        return {
            "added": 1,
            "modified": 0,
            "removed": 0,
            "unchanged": 2,
            "chunks": 1,
        }

    def ingest_file(self, file_path: str):
        return 3

    def rebuild(self, dir_path: str | None = None):
        return 5

    def delete_document(self, filename: str):
        return 1

    def get_documents(self):
        return [{"filename": "test.txt", "chunk_count": 3}]


@pytest.fixture
def client(monkeypatch):
    """返回已注入 Fake 依赖的 TestClient"""
    fake_executor = FakeTestExecutor()
    fake_pipeline = FakeRAGPipeline()

    # 覆盖 FastAPI Depends 注入
    app.dependency_overrides[get_test_executor] = lambda: fake_executor
    app.dependency_overrides[get_rag_pipeline] = lambda: fake_pipeline

    # 覆盖 lifespan 中的 RAG 预热与异步任务、知识库路由里的实例化
    monkeypatch.setattr("backend.api.deps.get_rag_pipeline", lambda: fake_pipeline)
    monkeypatch.setattr("backend.api.routes.knowledge.get_rag_pipeline", lambda: fake_pipeline)
    monkeypatch.setattr("backend.api.routes.testing.TestExecutorAgent", FakeTestExecutor)

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ---- 健康检查 ----

class TestHealthEndpoints:
    def test_health_lite(self, client):
        resp = client.get("/health/lite")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---- 自动化测试接口 ----

class TestTestingEndpoints:
    def test_run_tests_sync(self, client):
        payload = {
            "scenarios": ["login"],
            "base_url": "http://localhost:8000/demo",
            "headless": True,
            "timeout_seconds": 10,
            "auto_analyze": False,
            "sandbox": True,
        }
        resp = client.post("/api/v1/testing/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_scenarios"] == 1
        assert data["passed_count"] == 1
        assert data["results"][0]["scenario"] == "login"

    def test_run_tests_async(self, client):
        payload = {
            "scenarios": ["login"],
            "base_url": "http://localhost:8000/demo",
            "headless": True,
            "timeout_seconds": 10,
            "auto_analyze": False,
            "sandbox": True,
        }
        resp = client.post("/api/v1/testing/run/async", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        task_id = data["task_id"]

        # 异步任务在 TestClient 退出前会同步执行完成
        status_resp = client.get(f"/api/v1/testing/tasks/{task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] in ("completed", "running")

    def test_list_and_get_reports(self, client):
        # 先触发一次同步执行以产生报告
        payload = {
            "scenarios": ["login"],
            "base_url": "http://localhost:8000/demo",
            "sandbox": True,
        }
        client.post("/api/v1/testing/run", json=payload)

        list_resp = client.get("/api/v1/testing/reports")
        assert list_resp.status_code == 200
        reports = list_resp.json()["reports"]
        assert len(reports) >= 1

        report_id = reports[0]["id"]
        detail_resp = client.get(f"/api/v1/testing/reports/{report_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["id"] == report_id


# ---- 知识库接口 ----

class TestKnowledgeEndpoints:
    def test_knowledge_stats(self, client):
        resp = client.get("/api/v1/knowledge/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] == 10

    def test_knowledge_incremental(self, client):
        resp = client.post("/api/v1/knowledge/incremental")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["added"] == 1
        assert data["unchanged"] == 2

    def test_knowledge_documents(self, client):
        resp = client.get("/api/v1/knowledge/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] >= 0
        assert "documents" in data


# ---- Agent 对话记忆接口 ----

class TestAgentMemoryEndpoints:
    def test_clear_memory(self, client):
        resp = client.post("/api/v1/agent/memory/clear", json={"session_id": "test-sess"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["cleared"] is True
        assert data["session_id"] == "test-sess"

    def test_get_memory_history(self, client):
        # 先清空再验证空历史
        client.post("/api/v1/agent/memory/clear", json={"session_id": "history-sess"})
        resp = client.get("/api/v1/agent/memory/history-sess")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "history-sess"
        assert data["turn_count"] == 0
        assert data["history"] == []

    @patch("backend.api.routes.agents._execute_agent")
    def test_execute_saves_memory(self, mock_execute, client):
        """Agent 执行后应将用户输入和回复持久化到记忆存储"""
        mock_execute.return_value = {
            "task_type": "rag_query",
            "final_response": "这是测试回复",
            "tool_calls_made": 1,
            "iterations": 1,
            "error": "",
            "messages": [],
            "tool_results": [],
            "token_usage": [],
            "total_tokens": 0,
            "session_id": "mem-sess",
        }

        # mock 持久化存储，验证 add_turn 被调用
        mock_store = MagicMock()
        mock_store.format_context.return_value = ""
        mock_store.count_turns.return_value = 0

        with patch("backend.api.routes.agents._memory_store", mock_store):
            resp = client.post(
                "/api/v1/agent/execute",
                json={
                    "task": "测试问题",
                    "task_type": "auto",
                    "max_iterations": 3,
                    "session_id": "mem-sess",
                },
            )

        assert resp.status_code == 200
        mock_execute.assert_called_once()
        # 验证保存了本轮对话
        add_calls = mock_store.add_turn.call_args_list
        assert len(add_calls) == 1
        args, _ = add_calls[0]
        assert args[0] == "mem-sess"
        assert args[1] == "测试问题"
        assert args[2] == "这是测试回复"
