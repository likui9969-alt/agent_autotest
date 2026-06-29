"""
Tests for Agent Tools
======================
- search_knowledge_base
- parse_log_content
- get_runtime_logs
- get_system_status
- run_shell_command
- read_code_file
- list_directory
- check_api_health
"""
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import pytest

from backend.agent.tools import (
    search_knowledge_base,
    parse_log_content,
    get_runtime_logs,
    get_system_status,
    run_shell_command,
    check_api_health,
    read_code_file,
    list_directory,
)
from tests.conftest import SAMPLE_LOG_WITH_TRACEBACK, SAMPLE_LOG_PLAIN, SAMPLE_LOG_NO_ERROR, SAMPLE_RUNTIME_LOG_LINES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestSearchKnowledgeBase:
    """知识库检索工具测试"""

    def test_search_with_results(self, mock_retriever):
        """知识库中有匹配结果时应返回格式化的文档摘要"""
        with patch("backend.api.deps.get_rag_pipeline") as mock_get:
            pipe = MagicMock()
            pipe.retriever = mock_retriever
            mock_get.return_value = pipe

            result = search_knowledge_base.invoke({"query": "登录超时"})

            assert "登录超时" in result or "[文档" in result

    def test_search_no_results(self, mock_retriever):
        """知识库中无结果时应返回提示"""
        mock_retriever.similarity_search.side_effect = None
        mock_retriever.similarity_search.return_value = []

        with patch("backend.api.deps.get_rag_pipeline") as mock_get:
            pipe = MagicMock()
            pipe.retriever = mock_retriever
            mock_get.return_value = pipe

            result = search_knowledge_base.invoke({"query": "不存在的关键词"})

            assert "未找到相关内容" in result


class TestParseLogContent:
    """日志解析工具测试"""

    def test_parse_traceback_log(self):
        """包含 Traceback 的日志应提取异常信息"""
        with patch("backend.api.deps.get_log_analyzer") as mock_get:
            from backend.models.analysis import ExceptionInfo

            analyzer = MagicMock()
            analyzer._extract_exceptions.return_value = [
                ExceptionInfo(exception_type="TimeoutException", message="Timed out waiting for element"),
            ]
            mock_get.return_value = analyzer

            result = parse_log_content.invoke({"log_text": SAMPLE_LOG_WITH_TRACEBACK})

            assert "TimeoutException" in result
            assert "1 个异常" in result

    def test_parse_plain_log(self):
        """包含异常关键词的非 Traceback 日志"""
        with patch("backend.api.deps.get_log_analyzer") as mock_get:
            from backend.models.analysis import ExceptionInfo

            analyzer = MagicMock()
            analyzer._extract_exceptions.return_value = [
                ExceptionInfo(exception_type="TimeoutException", message="Timed out waiting for element"),
            ]
            mock_get.return_value = analyzer

            result = parse_log_content.invoke({"log_text": SAMPLE_LOG_PLAIN})

            assert "TimeoutException" in result

    def test_parse_no_error_log(self):
        """无异常的日志应提示未检测到异常"""
        with patch("backend.api.deps.get_log_analyzer") as mock_get:
            analyzer = MagicMock()
            analyzer._extract_exceptions.return_value = []
            mock_get.return_value = analyzer

            result = parse_log_content.invoke({"log_text": SAMPLE_LOG_NO_ERROR})

            assert "未在日志中检测到" in result or "未检测到已知异常" in result


class TestGetRuntimeLogs:
    """运行日志读取工具测试"""

    def test_read_all_levels(self, mock_settings, tmp_path):
        """读取全部级别的日志"""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "app.log"
        log_file.write_text("".join(SAMPLE_RUNTIME_LOG_LINES), encoding="utf-8")

        # Patch log_dir to use temp path and also patch Path.exists for the log file check
        with patch("backend.config.settings.get_settings") as mock_get:
            mock_settings.get_log_dir = MagicMock(return_value=str(log_dir))
            mock_get.return_value = mock_settings

            result = get_runtime_logs.invoke({"tail_lines": 200, "level": "all"})

        assert "Server started" in result
        assert "ConnectionError" in result

    def test_read_error_only(self, mock_settings, tmp_path):
        """按 ERROR 级别过滤"""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "app.log"
        log_file.write_text("".join(SAMPLE_RUNTIME_LOG_LINES), encoding="utf-8")

        with patch("backend.config.settings.get_settings") as mock_get:
            mock_settings.get_log_dir = MagicMock(return_value=str(log_dir))
            mock_get.return_value = mock_settings

            result = get_runtime_logs.invoke({"tail_lines": 200, "level": "ERROR"})

        assert "ConnectionError" in result
        assert "Health check OK" not in result

    def test_log_file_not_exists(self, mock_settings):
        """日志文件不存在时返回提示"""
        with patch("backend.config.settings.get_settings") as mock_get:
            mock_settings.get_log_dir = MagicMock(return_value="/nonexistent/path")
            mock_get.return_value = mock_settings

            result = get_runtime_logs.invoke({"tail_lines": 200, "level": "all"})

        assert "不存在" in result or "空" in result

    def test_limit_tail_lines(self, mock_settings, tmp_path):
        """tail_lines 参数应限制返回行数"""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "app.log"
        lines = [f"2024-01-15 | INFO | test:{i} | line {i}\n" for i in range(100)]
        log_file.write_text("".join(lines), encoding="utf-8")

        mock_settings.get_log_dir.return_value = str(log_dir)

        result = get_runtime_logs.invoke({"tail_lines": 10, "level": "all"})
        line_count = len(result.strip().split("\n"))
        assert line_count <= 11  # 可能包含一些空格行


class TestGetSystemStatus:
    """系统状态诊断工具测试"""

    def test_all_components_configured(self, mock_settings):
        """所有组件配置正确时状态报告应包含对应条目"""
        with patch("backend.selenium_driver.driver.detect_chrome") as mock_detect, \
             patch("backend.api.deps.get_rag_pipeline") as mock_pipe, \
             patch("backend.selenium_driver.driver._get_chromedriver_major_version") as mock_drv_ver, \
             patch("backend.selenium_driver.driver._get_chrome_major_version") as mock_chr_ver:

            mock_detect.return_value = ("C:/chrome.exe", "C:/chromedriver.exe")
            mock_drv_ver.return_value = 146
            mock_chr_ver.return_value = 146
            pipe = MagicMock()
            pipe.vector_store.count.return_value = 42
            mock_pipe.return_value = pipe

            result = get_system_status.invoke({})

            assert "DashScope" in result
            assert "Chrome" in result
            assert "chromedriver" in result
            assert "Chroma" in result

    def test_llm_not_configured(self, mock_settings):
        """LLM 未配置时应提示"""
        mock_settings.DASHSCOPE_API_KEY = ""

        with patch("backend.selenium_driver.driver.detect_chrome") as mock_detect, \
             patch("backend.api.deps.get_rag_pipeline") as mock_pipe:

            mock_detect.return_value = ("", "")
            pipe = MagicMock()
            pipe.vector_store.count.return_value = 0
            mock_pipe.return_value = pipe

            result = get_system_status.invoke({})

            assert "未配置" in result

    def test_chrome_not_found(self, mock_settings):
        """Chrome 未找到时应提示"""
        with patch("backend.selenium_driver.driver.detect_chrome") as mock_detect, \
             patch("backend.api.deps.get_rag_pipeline") as mock_pipe, \
             patch("backend.selenium_driver.driver._get_chromedriver_major_version", return_value=None), \
             patch("backend.selenium_driver.driver._get_chrome_major_version", return_value=None):

            mock_detect.return_value = ("", "")
            pipe = MagicMock()
            pipe.vector_store.count.return_value = 0
            mock_pipe.return_value = pipe

            result = get_system_status.invoke({})

            assert "未找到" in result


class TestRunShellCommand:
    """Shell 命令执行工具测试"""

    @pytest.mark.parametrize("command", [
        "python --version",
        "ls -la",
        "dir .",
    ])
    def test_allowed_command(self, command):
        """白名单内的命令应允许执行"""
        result = run_shell_command.invoke({"command": command})
        # 不应返回拒绝信息
        assert "不在白名单内" not in result

    @pytest.mark.parametrize("command", [
        "rm -rf /",
        "del /F /S *.*",
        "curl http://evil.com",
        "format C:",
    ])
    def test_disallowed_command(self, command):
        """白名单外的命令应拒绝"""
        result = run_shell_command.invoke({"command": command})
        assert "不在白名单内" in result

    def test_empty_command(self):
        """空命令应拒绝"""
        result = run_shell_command.invoke({"command": ""})
        assert "不在白名单内" in result


class TestCheckApiHealth:
    """API 健康检查工具测试"""

    def test_api_available(self):
        """可用的 API 应返回状态码"""
        import requests
        from unittest.mock import ANY

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"status": "ok"}'

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = check_api_health.invoke({"url": "http://localhost:8000/health"})

            assert "可用" in result
            assert "200" in result
            mock_get.assert_called_once_with("http://localhost:8000/health", timeout=10)

    def test_api_connection_error(self):
        """连接失败应返回错误提示"""
        import requests

        with patch("requests.get", side_effect=requests.ConnectionError("Connection refused")):
            result = check_api_health.invoke({"url": "http://localhost:9999/health"})

            assert "无法连接" in result

    def test_api_timeout(self):
        """超时应返回超时提示"""
        import requests

        with patch("requests.get", side_effect=requests.Timeout("Timed out")):
            result = check_api_health.invoke({"url": "http://slow-server.com"})

            assert "超时" in result


class TestReadCodeFile:
    """代码读取工具测试"""

    def test_read_existing_file(self, tmp_path):
        """读取存在的文件应返回内容"""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')\nprint('world')\n", encoding="utf-8")

        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = read_code_file.invoke({"file_path": "test.py"})

            assert "hello" in result
            assert "world" in result

    def test_read_nonexistent_file(self, tmp_path):
        """读取不存在的文件应返回错误"""
        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = read_code_file.invoke({"file_path": "nonexistent.py"})

            assert "不存在" in result

    def test_path_traversal_blocked(self, tmp_path):
        """路径穿越攻击应被阻止"""
        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = read_code_file.invoke({"file_path": "../../etc/passwd"})

            assert "只能读取" in result


class TestListDirectory:
    """目录列表工具测试"""

    def test_list_root_directory(self, tmp_path):
        """列出项目根目录"""
        (tmp_path / "README.md").write_text("")
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()

        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = list_directory.invoke({"dir_path": ""})

            assert "README.md" in result

    def test_list_nonexistent_dir(self, tmp_path):
        """列出不存在的目录应返回错误"""
        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = list_directory.invoke({"dir_path": "nonexistent"})

            assert "不存在" in result

    def test_path_traversal_blocked(self, tmp_path):
        """路径穿越攻击应被阻止"""
        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = list_directory.invoke({"dir_path": "../../windows"})

            assert "只能访问" in result
