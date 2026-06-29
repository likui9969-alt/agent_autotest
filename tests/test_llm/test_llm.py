"""
Tests for LLM Module
====================
- LLMClient: chat, embed, chat_with_tools
- Prompt templates: formatting, variable injection
- Retry mechanism: exponential backoff
"""
from unittest.mock import patch, MagicMock, ANY
import httpx
import pytest
from openai import (
    APIStatusError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)

from backend.llm.client import LLMClient


def _make_error_response(status_code: int) -> httpx.Response:
    """构造模拟的 httpx.Response 用于 OpenAI 异常"""
    return httpx.Response(status_code=status_code, request=httpx.Request("POST", "https://test"))


class TestLLMClientChat:
    """LLMClient.chat 测试"""

    def test_chat_normal_response(self):
        """正常的 chat 调用应返回字符串"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "这是一个测试回复。"
            mock_completion.choices = [mock_choice]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.chat([
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "你好"},
            ])

            assert result == "这是一个测试回复。"

    def test_chat_empty_response(self):
        """空回复应返回空字符串"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = ""
            mock_completion.choices = [mock_choice]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.chat([
                {"role": "user", "content": "test"},
            ])

            assert result == ""

    def test_chat_with_temperature_override(self):
        """temperature 覆盖应传递给 API"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_completion = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "ok"
            mock_completion.choices = [mock_choice]
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            client.chat([{"role": "user", "content": "hi"}], temperature=0.5)

            mock_client.chat.completions.create.assert_called_with(
                model=ANY,
                messages=ANY,
                temperature=0.5,
                max_tokens=ANY,
                stream=ANY,
            )

    def test_chat_streaming(self):
        """流式模式应返回迭代器"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_chunk1 = MagicMock()
            type(mock_chunk1).choices = [MagicMock(delta=MagicMock(content="Hello"))]

            mock_chunk2 = MagicMock()
            type(mock_chunk2).choices = [MagicMock(delta=MagicMock(content=" World"))]

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.chat([{"role": "user", "content": "hi"}], stream=True)

            tokens = list(result)
            assert tokens == ["Hello", " World"]

    def test_chat_api_error(self):
        """API 调用失败应传播异常"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            client = LLMClient()
            with pytest.raises(Exception, match="API Error"):
                client.chat([{"role": "user", "content": "test"}])


class TestLLMClientEmbed:
    """LLMClient.embed 测试"""

    def test_embed_normal(self):
        """正常的嵌入调用应返回向量列表"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_data = [MagicMock(index=0, embedding=[0.1, 0.2, 0.3]),
                         MagicMock(index=1, embedding=[0.4, 0.5, 0.6])]
            mock_response = MagicMock()
            mock_response.data = mock_data

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.embed(["text1", "text2"])

            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]

    def test_embed_empty_list(self):
        """空列表应返回空列表"""
        with patch("backend.llm.client.OpenAI"):
            client = LLMClient()
            result = client.embed([])
            assert result == []

    def test_embed_returns_sorted_by_index(self):
        """嵌入结果应按 index 排序"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_data = [MagicMock(index=1, embedding=[0.4]),
                         MagicMock(index=0, embedding=[0.1])]
            mock_response = MagicMock()
            mock_response.data = mock_data

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.embed(["a", "b"])
            assert result == [[0.1], [0.4]]

    def test_embed_single(self):
        """embed_single 应返回单个向量"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_data = [MagicMock(index=0, embedding=[0.1, 0.2])]
            mock_response = MagicMock()
            mock_response.data = mock_data

            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.embed_single("test")
            assert result == [0.1, 0.2]


class TestLLMClientChatWithTools:
    """LLMClient.chat_with_tools 测试"""

    def test_no_tool_call(self):
        """无工具调用时应返回 content 空 tool_calls"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_msg = MagicMock()
            mock_msg.content = "最终回答"
            mock_msg.tool_calls = None

            mock_completion = MagicMock()
            mock_completion.choices[0].message = mock_msg

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.chat_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "test", "description": "", "parameters": {}}}],
            )

            assert result["content"] == "最终回答"
            assert result["tool_calls"] is None

    def test_with_tool_call(self):
        """有工具调用时应正确解析"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            # 创建一个模拟的 tool call
            mock_tc = MagicMock()
            mock_tc.id = "call_abc123"
            mock_tc.function.name = "search_knowledge_base"
            mock_tc.function.arguments = '{"query": "登录超时"}'

            mock_msg = MagicMock()
            mock_msg.content = ""
            mock_msg.tool_calls = [mock_tc]

            mock_completion = MagicMock()
            mock_completion.choices[0].message = mock_msg

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.chat_with_tools(
                messages=[{"role": "user", "content": "搜索知识库"}],
                tools=[{"type": "function", "function": {"name": "search_knowledge_base", "description": "", "parameters": {}}}],
            )

            assert result["content"] == ""
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["name"] == "search_knowledge_base"
            assert result["tool_calls"][0]["args"]["query"] == "登录超时"

    def test_invalid_json_in_tool_args(self):
        """工具参数 JSON 无效时应返回空字典"""
        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_tc = MagicMock()
            mock_tc.id = "call_bad"
            mock_tc.function.name = "test"
            mock_tc.function.arguments = "{invalid json}"

            mock_msg = MagicMock()
            mock_msg.content = ""
            mock_msg.tool_calls = [mock_tc]

            mock_completion = MagicMock()
            mock_completion.choices[0].message = mock_msg

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            client = LLMClient()
            result = client.chat_with_tools(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
            )

            assert result["tool_calls"][0]["args"] == {}


class TestPromptTemplates:
    """Prompt 模板测试"""

    def test_get_template_exists(self):
        """获取存在的模板应返回 PromptTemplate"""
        from backend.llm.prompts import get_template

        for name in ["rag_query", "log_analysis", "test_generation", "jira_creation"]:
            tmpl = get_template(name)
            assert tmpl.name is not None
            assert tmpl.system
            assert tmpl.user
            assert 0 <= tmpl.temperature <= 2

    def test_get_template_nonexistent(self):
        """获取不存在的模板应抛 KeyError"""
        from backend.llm.prompts import get_template

        with pytest.raises(KeyError):
            get_template("nonexistent_template")

    def test_rag_query_template_formatting(self):
        """RAG 问答模板应正确填入变量"""
        from backend.llm.prompts import get_template

        tmpl = get_template("rag_query")
        formatted = tmpl.user.format(question="登录超时怎么办", context="相关知识库内容")

        assert "登录超时怎么办" in formatted
        assert "相关知识库内容" in formatted

    def test_log_analysis_template_formatting(self):
        """日志分析模板应正确填入变量"""
        from backend.llm.prompts import get_template

        tmpl = get_template("log_analysis")
        formatted = tmpl.user.format(log_content="ERROR: Timeout", historical_cases="Case #1")

        assert "ERROR: Timeout" in formatted
        assert "Case #1" in formatted

    def test_all_templates_registered(self):
        """所有模板应注册在 TEMPLATES 字典中"""
        from backend.llm.prompts import TEMPLATES

        assert len(TEMPLATES) >= 4
        assert "rag_query" in TEMPLATES
        assert "log_analysis" in TEMPLATES
        assert "test_generation" in TEMPLATES
        assert "jira_creation" in TEMPLATES


class TestLLMRetry:
    """LLM 调用重试机制测试"""

    def test_retry_transient_then_success(self):
        """首次失败（httpx 超时），第二次成功"""
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "重试成功"
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            httpx.ReadTimeout("timeout"),
            mock_completion,
        ]

        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_openai.return_value = mock_client
            client = LLMClient()
            result = client.chat([{"role": "user", "content": "hi"}])

        assert result == "重试成功"
        assert mock_client.chat.completions.create.call_count == 2

    def test_retry_all_fail(self):
        """所有重试都失败，最终抛出异常"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = httpx.ReadTimeout("persistent")

        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_openai.return_value = mock_client

            client = LLMClient()
            with pytest.raises(Exception):
                client.chat([{"role": "user", "content": "hi"}])

            assert mock_client.chat.completions.create.call_count == 3

    def test_no_retry_on_non_retryable(self):
        """401 等异常不重试，直接抛出"""
        mock_client = MagicMock()
        resp = httpx.Response(401, request=httpx.Request("POST", "https://test"))
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="Invalid API key",
            response=resp,
            body=None,
        )

        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_openai.return_value = mock_client
            client = LLMClient()

            with pytest.raises(APIStatusError):
                client.chat([{"role": "user", "content": "hi"}])

            assert mock_client.chat.completions.create.call_count == 1

    def test_retry_embed_transient_failure(self):
        """embed 的可重试异常也应触发重试"""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(index=0, embedding=[0.1])]
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = [
            httpx.ConnectError("connection lost"),
            mock_response,
        ]

        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_openai.return_value = mock_client
            client = LLMClient()
            result = client.embed(["text"])

        assert len(result) == 1
        assert mock_client.embeddings.create.call_count == 2

    def test_retry_chat_with_tools(self):
        """chat_with_tools 的可重试异常也应触发重试"""
        mock_msg = MagicMock()
        mock_msg.content = "最终回答"
        mock_msg.tool_calls = None
        mock_completion = MagicMock()
        mock_completion.choices[0].message = mock_msg

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            httpx.RemoteProtocolError("server reset"),
            mock_completion,
        ]

        with patch("backend.llm.client.OpenAI") as mock_openai:
            mock_openai.return_value = mock_client
            client = LLMClient()

            result = client.chat_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
            )

        assert result["content"] == "最终回答"
        assert mock_client.chat.completions.create.call_count == 2
