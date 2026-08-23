"""
API 集成测试
============
覆盖 /api/v1/testing 和 /api/v1/knowledge 的核心端点，
通过依赖注入覆盖避免真实调用 Selenium / Chroma / LLM。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.deps import get_rag_pipeline, get_test_executor
from backend.main import app
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


# ---- 认证中间件（fail-closed）----

def _make_settings(api_token: str = "", debug: bool = True):
    """构造可控的 settings mock（控制认证行为）"""
    fake = MagicMock()
    fake.API_TOKEN = api_token
    fake.DEBUG = debug
    return fake


class TestAuthMiddleware:
    """认证中间件测试（2026-08-23 安全收口）

    策略验证：
    - 配置 token：非白名单请求需 Bearer Token
    - 未配置 token + 生产模式（DEBUG=False）：写操作 503 fail-closed
    - 未配置 token + 开发模式（DEBUG=True）：放行
    """

    def test_no_token_production_write_blocked(self, client):
        """生产模式 + 未配置 token：写操作必须被拒绝（fail-closed）"""
        with patch("backend.api.auth.get_settings", return_value=_make_settings("", debug=False)):
            resp = client.post("/api/v1/rag/query", json={"question": "test"})
            assert resp.status_code == 503
            assert "API_TOKEN" in resp.json()["message"]

    def test_no_token_production_read_allowed(self, client):
        """生产模式 + 未配置 token：只读 GET 放行"""
        with patch("backend.api.auth.get_settings", return_value=_make_settings("", debug=False)):
            resp = client.get("/api/v1/knowledge/stats")
            assert resp.status_code == 200

    @pytest.mark.parametrize("path", [
        "/api/v1/agent/memory/test-session",
        "/api/v1/agent/reports",
        "/api/v1/agent/reports/report-001",
        "/api/v1/knowledge/documents",
    ])
    def test_no_token_production_sensitive_read_blocked(self, client, path):
        """生产模式 + 未配置 token：敏感只读路径（对话历史/报告/文档）也应拒绝（P2-2 修复）"""
        with patch("backend.api.auth.get_settings", return_value=_make_settings("", debug=False)):
            resp = client.get(path)
            assert resp.status_code == 503

    def test_no_token_dev_write_allowed(self, client):
        """开发模式 + 未配置 token：写操作放行（本地调试便利）"""
        with patch("backend.api.auth.get_settings", return_value=_make_settings("", debug=True)):
            resp = client.post("/api/v1/agent/memory/clear", json={"session_id": "s"})
            assert resp.status_code == 200

    def test_token_configured_missing_header_rejected(self, client):
        """配置 token 后：缺失 Authorization 头应 401"""
        with patch("backend.api.auth.get_settings", return_value=_make_settings("secret-token")):
            resp = client.get("/api/v1/knowledge/stats")
            assert resp.status_code == 401

    def test_token_configured_wrong_token_rejected(self, client):
        """配置 token 后：错误 token 应 401"""
        with patch("backend.api.auth.get_settings", return_value=_make_settings("secret-token")):
            resp = client.get(
                "/api/v1/knowledge/stats",
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert resp.status_code == 401

    def test_token_configured_correct_token_allowed(self, client):
        """配置 token 后：正确 token 应放行"""
        with patch("backend.api.auth.get_settings", return_value=_make_settings("secret-token")):
            resp = client.get(
                "/api/v1/knowledge/stats",
                headers={"Authorization": "Bearer secret-token"},
            )
            assert resp.status_code == 200

    def test_health_path_always_whitelisted(self, client):
        """健康检查路径永远免认证（含生产模式无 token 场景）"""
        with patch("backend.api.auth.get_settings", return_value=_make_settings("", debug=False)):
            resp = client.get("/health/lite")
            assert resp.status_code == 200


# ---- 上传路径穿越防护 ----

class TestUploadPathTraversal:
    """上传接口路径穿越测试（2026-08-23 安全收口）"""

    def test_traversal_filename_sanitized(self, client, tmp_path):
        """../../evil.txt 应被清洗为 evil.txt，保存在上传目录内"""
        fake_settings = MagicMock()
        fake_settings.get_upload_dir = lambda: str(tmp_path)

        with patch("backend.api.routes.knowledge.get_settings", return_value=fake_settings):
            resp = client.post(
                "/api/v1/knowledge/upload",
                files={"file": ("../../evil.txt", b"hello world", "text/plain")},
            )

        assert resp.status_code == 200
        # 文件只应出现在上传目录第一层（basename 清洗，无子目录穿越）
        saved = list(tmp_path.iterdir())
        assert len(saved) == 1
        assert saved[0].is_file()
        assert saved[0].name == "evil.txt"
        saved[0].unlink(missing_ok=True)

    def test_windows_style_traversal_sanitized(self, client, tmp_path):
        """Windows 风格路径 ..\\..\\evil2.txt 不应写出上传目录"""
        import sys

        fake_settings = MagicMock()
        fake_settings.get_upload_dir = lambda: str(tmp_path)

        with patch("backend.api.routes.knowledge.get_settings", return_value=fake_settings):
            resp = client.post(
                "/api/v1/knowledge/upload",
                files={"file": ("..\\..\\evil2.txt", b"data", "text/plain")},
            )

        assert resp.status_code == 200
        # 无论如何，文件只能出现在上传目录内（穿越被阻止）
        saved = list(tmp_path.iterdir())
        assert len(saved) == 1
        assert saved[0].is_file()
        name = saved[0].name
        # 文件名不含正斜杠（任何平台都不允许目录注入）
        assert "/" not in name
        # Windows 上反斜杠是路径分隔符，应被剥离
        if sys.platform == "win32":
            assert name == "evil2.txt"
        saved[0].unlink(missing_ok=True)


# ---- JIRA 集成端点（依赖注入 mock，不访问真实 JIRA）----

class FakeJiraCreator:
    """模拟 JiraCreator 的三种状态（success / skipped / failed）"""

    def __init__(self, status: str = "success"):
        self._status = status

    def check_connection(self):
        if self._status == "success":
            return {"status": "connected", "message": "连接成功（JIRA 1001.0.0）"}
        return {"status": "unconfigured", "message": "JIRA 未配置（缺少 JIRA_URL）"}

    def create_issue(self, request):
        from backend.models.jira import JiraCreateResponse
        if self._status == "success":
            return JiraCreateResponse(
                status="success",
                issue_key="KAN-7",
                issue_url="https://example.atlassian.net/browse/KAN-7",
                message="缺陷单创建成功",
            )
        if self._status == "skipped":
            return JiraCreateResponse(
                status="skipped",
                issue_key="",
                issue_url="",
                message="JIRA 未配置（缺少 JIRA_URL），无法创建缺陷单。",
            )
        return JiraCreateResponse(status="failed", message="JIRA API 错误 (401): 认证失败")


class TestJiraEndpoints:
    """JIRA API E2E 测试（2026-08-23：JIRA 链路跑通后补齐）

    通过 dependency_overrides 注入 FakeJiraCreator，
    覆盖 建单成功 / 未配置跳过 / 失败 / 连接状态 四条路径。
    """

    def _client_with(self, monkeypatch, fake_creator):
        from backend.api.deps import get_jira_creator
        app.dependency_overrides[get_jira_creator] = lambda: fake_creator
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.pop(get_jira_creator, None)

    def test_create_success(self, client, monkeypatch):
        payload = {
            "title": "登录页面加载超时（>30s）",
            "description": "用户在点击登录按钮后，页面加载超过30秒无响应",
            "priority": "High",
            "log_content": "TimeoutException: Page load timeout...",
            "ai_analysis": "可能原因：后端接口响应慢",
        }
        gen = self._client_with(monkeypatch, FakeJiraCreator("success"))
        c = next(gen)
        resp = c.post("/api/v1/jira/create", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["issue_key"] == "KAN-7"
        assert "browse/KAN-7" in data["issue_url"]
        try:
            next(gen)
        except StopIteration:
            pass

    def test_create_unconfigured_skipped(self, client, monkeypatch):
        """JIRA 未配置时应返回 skipped（优雅降级，不抛异常）"""
        gen = self._client_with(monkeypatch, FakeJiraCreator("skipped"))
        c = next(gen)
        resp = c.post(
            "/api/v1/jira/create",
            json={"title": "这是一条测试缺陷单", "description": "这是一个测试缺陷描述"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"
        assert "未配置" in resp.json()["message"]
        try:
            next(gen)
        except StopIteration:
            pass

    def test_create_api_failure(self, client, monkeypatch):
        """JIRA API 失败时应返回 failed + 错误信息（不抛异常）"""
        gen = self._client_with(monkeypatch, FakeJiraCreator("failed"))
        c = next(gen)
        resp = c.post(
            "/api/v1/jira/create",
            json={"title": "这是一条测试缺陷单", "description": "这是一个测试缺陷描述"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
        try:
            next(gen)
        except StopIteration:
            pass

    def test_status_connected(self, client, monkeypatch):
        gen = self._client_with(monkeypatch, FakeJiraCreator("success"))
        c = next(gen)
        resp = c.get("/api/v1/jira/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "connected"
        try:
            next(gen)
        except StopIteration:
            pass


class TestJiraIssueTypeResolution:
    """JiraCreator._resolve_issue_type 单元测试（2026-08-23 动态类型解析）

    背景：team-managed 项目（如中文模板）可能没有 "Bug" 类型，
    写死会 400。解析逻辑按候选优先级动态选择。
    """

    def _make_creator(self, project_key="KAN"):
        from backend.agent.jira_creator import JiraCreator
        creator = JiraCreator.__new__(JiraCreator)  # 跳过 __init__（不依赖 LLM）
        creator.settings = MagicMock()
        creator.settings.JIRA_PROJECT_KEY = project_key
        # 每个用例独立缓存，避免类级缓存串扰
        JiraCreator._issue_type_cache.pop(project_key, None)
        return creator

    def _mock_client(self, status_code=200, issue_types=None):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {"issueTypes": issue_types or []}
        mock_client.get.return_value = mock_resp
        return mock_client

    def test_selects_bug_when_available(self):
        creator = self._make_creator()
        client = self._mock_client(issue_types=[{"name": "Bug"}, {"name": "Task"}])
        assert creator._resolve_issue_type(client, "https://x.atlassian.net", ()) == "Bug"

    def test_falls_back_to_chinese_task_type(self):
        """中文模板无 Bug 类型时应选 '任务'（候选优先级：Bug > 故障 > 任务）"""
        creator = self._make_creator()
        client = self._mock_client(issue_types=[
            {"name": "长篇故事", "subtask": False},
            {"name": "Subtask", "subtask": True},
            {"name": "任务", "subtask": False},
            {"name": "故事", "subtask": False},
        ])
        assert creator._resolve_issue_type(client, "https://x.atlassian.net", ()) == "任务"

    def test_falls_back_to_english_task_type(self):
        creator = self._make_creator()
        client = self._mock_client(issue_types=[{"name": "Task"}, {"name": "Epic"}])
        assert creator._resolve_issue_type(client, "https://x.atlassian.net", ()) == "Task"

    def test_unexpected_types_use_first_non_subtask(self):
        """全候选未命中时用第一个非 subtask 类型兜底"""
        creator = self._make_creator()
        client = self._mock_client(issue_types=[
            {"name": "Custom Type", "subtask": False},
            {"name": "Sub", "subtask": True},
        ])
        assert creator._resolve_issue_type(client, "https://x.atlassian.net", ()) == "Custom Type"

    def test_meta_failure_falls_back_to_bug(self):
        """createmeta 查询失败（网络/404）时回退默认 'Bug'"""
        creator = self._make_creator()
        client = self._mock_client(status_code=404)
        assert creator._resolve_issue_type(client, "https://x.atlassian.net", ()) == "Bug"

    def test_result_is_cached(self):
        """解析结果应写入类级缓存，第二次调用不再发请求"""
        creator = self._make_creator()
        client = self._mock_client(issue_types=[{"name": "Bug"}])
        first = creator._resolve_issue_type(client, "https://x.atlassian.net", ())
        assert first == "Bug"
        assert client.get.call_count == 1
        # 第二次走缓存，不再请求
        assert creator._resolve_issue_type(client, "https://x.atlassian.net", ()) == "Bug"
        assert client.get.call_count == 1
        # 清理缓存，避免影响其他用例
        from backend.agent.jira_creator import JiraCreator
        JiraCreator._issue_type_cache.pop("KAN", None)
