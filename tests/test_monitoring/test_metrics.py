"""
可观测性模块测试
================
- MetricsMiddleware：HTTP 请求计数/耗时记录
- /metrics 端点：Prometheus 文本协议输出
- /api/v1/metrics/tokens：Token 用量 JSON 查询
- TokenUsageTracker：聚合与快照
- Provider 埋点：chat / chat_with_tools / embed 的 token 记录
- LLMClient 埋点：调用计数（含流式）

注意：Prometheus 计数器是进程级全局的，测试一律用"前后差值"断言，
不依赖绝对值。
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from backend.llm.client import LLMClient
from backend.main import app
from backend.monitoring.token_tracker import token_tracker


def _counter_value(name: str, labels: dict | None = None) -> float:
    """读取 Prometheus 计数器当前值（指标不存在返回 0）"""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _make_settings(api_token: str = "", debug: bool = True):
    """构造可控的 settings mock（控制认证行为，同 test_routes.py 模式）"""
    fake = MagicMock()
    fake.API_TOKEN = api_token
    fake.DEBUG = debug
    return fake


@pytest.fixture
def client(monkeypatch):
    """轻量 TestClient — 屏蔽 lifespan 的真实 RAG 预热"""
    monkeypatch.setattr("backend.api.deps.get_rag_pipeline", MagicMock())
    with TestClient(app) as c:
        yield c


# ==================== /metrics 端点与 HTTP 中间件 ====================

class TestMetricsEndpoint:
    """/metrics 端点测试"""

    def test_metrics_endpoint_returns_prometheus_text(self, client):
        """/metrics 应返回 200 与 Prometheus 文本协议"""
        resp = client.get("/metrics")

        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        # 默认 Python 运行时指标必然存在（跨平台稳定）
        assert "python_info" in resp.text
        # 自定义指标已注册（HELP 文本总会输出，样本可有可无）
        assert "http_requests_total" in resp.text
        assert "llm_calls_total" in resp.text
        assert "llm_tokens_total" in resp.text

    def test_http_metrics_records_requests(self, client):
        """HTTP 中间件应按路由模板记录请求计数"""
        before = _counter_value(
            "http_requests_total",
            {"method": "GET", "path": "/health/lite", "status": "200"},
        )

        resp = client.get("/health/lite")
        assert resp.status_code == 200

        after = _counter_value(
            "http_requests_total",
            {"method": "GET", "path": "/health/lite", "status": "200"},
        )
        assert after == before + 1

    def test_metrics_endpoint_no_auth_required(self):
        """/metrics 在认证中间件白名单内（无 token 可访问）"""
        with patch(
            "backend.api.auth.get_settings",
            return_value=_make_settings("secret-token", debug=False),
        ), TestClient(app) as c:
            resp = c.get("/metrics")
            assert resp.status_code == 200

    def test_unmatched_path_uses_unmatched_label(self, client):
        """404 请求的 path 标签归入 unmatched（基数受控）"""
        before = _counter_value(
            "http_requests_total",
            {"method": "GET", "path": "unmatched", "status": "404"},
        )

        resp = client.get("/definitely-not-a-route-xyz")
        assert resp.status_code == 404

        after = _counter_value(
            "http_requests_total",
            {"method": "GET", "path": "unmatched", "status": "404"},
        )
        assert after == before + 1


# ==================== Token 追踪器 ====================

class TestTokenUsageTracker:
    """TokenUsageTracker 聚合测试"""

    def setup_method(self):
        token_tracker.reset()

    def test_record_and_snapshot(self):
        """record 应正确聚合，snapshot 应返回汇总与明细"""
        token_tracker.record("dashscope", "chat", prompt_tokens=100, completion_tokens=50)
        token_tracker.record("dashscope", "chat", prompt_tokens=30, completion_tokens=20)
        token_tracker.record("ollama", "embed", prompt_tokens=10)

        snap = token_tracker.snapshot()

        assert snap["totals"] == {
            "calls": 3,
            "prompt_tokens": 140,
            "completion_tokens": 70,
            "total_tokens": 210,
        }
        by_op = {(i["provider"], i["operation"]): i for i in snap["by_operation"]}
        assert by_op[("dashscope", "chat")]["total_tokens"] == 200
        assert by_op[("ollama", "embed")]["total_tokens"] == 10

    def test_record_ignores_non_positive(self):
        """非正数用量不记录（Mock 桩防御）"""
        token_tracker.record("dashscope", "chat", 0, 0)
        token_tracker.record("dashscope", "chat")

        assert token_tracker.snapshot()["totals"]["calls"] == 0

    def test_record_updates_prometheus_counter(self):
        """record 应同步上报 llm_tokens_total 指标"""
        before_prompt = _counter_value(
            "llm_tokens_total",
            {"provider": "testprov", "operation": "chat", "type": "prompt"},
        )
        before_completion = _counter_value(
            "llm_tokens_total",
            {"provider": "testprov", "operation": "chat", "type": "completion"},
        )

        token_tracker.record("testprov", "chat", prompt_tokens=7, completion_tokens=3)

        assert _counter_value(
            "llm_tokens_total",
            {"provider": "testprov", "operation": "chat", "type": "prompt"},
        ) == before_prompt + 7
        assert _counter_value(
            "llm_tokens_total",
            {"provider": "testprov", "operation": "chat", "type": "completion"},
        ) == before_completion + 3

    def test_tokens_json_endpoint(self, client):
        """GET /api/v1/metrics/tokens 应返回追踪器快照"""
        token_tracker.record("dashscope", "chat", prompt_tokens=42, completion_tokens=8)

        resp = client.get("/api/v1/metrics/tokens")

        assert resp.status_code == 200
        data = resp.json()
        assert data["totals"]["total_tokens"] >= 50
        assert any(
            i["provider"] == "dashscope" and i["operation"] == "chat"
            for i in data["by_operation"]
        )


# ==================== Provider / Client 埋点 ====================

def _usage_mock(prompt: int, completion: int) -> MagicMock:
    """构造真实整数字段的 usage 对象"""
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    return u


class TestProviderTokenRecording:
    """OpenAICompatibleProvider 的 token 埋点测试"""

    def setup_method(self):
        token_tracker.reset()

    def test_chat_records_tokens(self):
        """非流式 chat 应记录 usage token"""
        with patch("backend.llm.providers.openai_compatible.OpenAI") as mock_openai:
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_completion.usage = _usage_mock(120, 45)
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            client.chat([{"role": "user", "content": "hi"}])

            snap = token_tracker.snapshot()
            item = next(
                i for i in snap["by_operation"] if i["operation"] == "chat"
            )
            assert item["prompt_tokens"] == 120
            assert item["completion_tokens"] == 45

    def test_chat_mock_usage_not_recorded(self):
        """Mock 桩的 usage（非整数字段）不应污染 token 统计"""
        with patch("backend.llm.providers.openai_compatible.OpenAI") as mock_openai:
            mock_completion = MagicMock()  # usage 是 MagicMock，字段也是 MagicMock
            mock_completion.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            client.chat([{"role": "user", "content": "hi"}])

            snap = token_tracker.snapshot()
            assert snap["totals"]["calls"] == 0

    def test_chat_with_tools_records_tokens(self):
        """chat_with_tools 应记录 usage token"""
        with patch("backend.llm.providers.openai_compatible.OpenAI") as mock_openai:
            msg = MagicMock()
            msg.content = "分析中"
            msg.tool_calls = None
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock(message=msg)]
            mock_completion.usage = _usage_mock(200, 60)
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.chat_with_tools(
                [{"role": "user", "content": "分析日志"}], tools=[]
            )

            assert result["content"] == "分析中"
            snap = token_tracker.snapshot()
            item = next(
                i for i in snap["by_operation"] if i["operation"] == "chat_with_tools"
            )
            assert item["prompt_tokens"] == 200
            assert item["completion_tokens"] == 60


class TestClientCallMetrics:
    """LLMClient 调用计数指标测试"""

    def test_chat_success_increments_counter(self):
        """成功的 chat 调用应递增 llm_calls_total{status=success}"""
        before = _counter_value(
            "llm_calls_total", {"provider": "dashscope", "operation": "chat", "status": "success"}
        )

        with patch("backend.llm.providers.openai_compatible.OpenAI") as mock_openai:
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_completion.usage = None
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            client.chat([{"role": "user", "content": "hi"}])

        after = _counter_value(
            "llm_calls_total", {"provider": "dashscope", "operation": "chat", "status": "success"}
        )
        assert after == before + 1

    def test_chat_stream_records_on_consumption(self):
        """流式 chat 在消费完成后应递增 success 计数"""
        before = _counter_value(
            "llm_calls_total", {"provider": "dashscope", "operation": "chat", "status": "success"}
        )

        with patch("backend.llm.providers.openai_compatible.OpenAI") as mock_openai:
            chunk = MagicMock()
            type(chunk).choices = [MagicMock(delta=MagicMock(content="Hi"))]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter([chunk])
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.chat([{"role": "user", "content": "hi"}], stream=True)

            # 未消费时不计数（generator 延迟执行）
            mid = _counter_value(
                "llm_calls_total",
                {"provider": "dashscope", "operation": "chat", "status": "success"},
            )
            assert mid == before

            tokens = list(result)
            assert tokens == ["Hi"]

        after = _counter_value(
            "llm_calls_total", {"provider": "dashscope", "operation": "chat", "status": "success"}
        )
        assert after == before + 1
