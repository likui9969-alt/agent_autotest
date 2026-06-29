"""
Agent 工具定义模块
使用 LangChain @tool 装饰器，将后端能力封装为 Agent 可调用的工具函数
"""
import logging
import os
import sys
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger("ai_rd_agent")


# ==================== 工具 1：知识库检索 ====================

@tool
def search_knowledge_base(query: str) -> str:
    """在向量知识库中搜索与查询相关的内容。

    使用场景：
    - 需要了解历史故障案例时
    - 需要查找技术文档中的解决方案时
    - 需要检索类似问题的处理方法时

    Args:
        query: 搜索查询，例如 "登录超时如何解决" 或 "数据库连接池耗尽"

    Returns:
        检索到的相关文档内容（最多 5 条，按相似度排序）
    """
    from backend.api.deps import get_rag_pipeline

    pipeline = get_rag_pipeline()
    docs = pipeline.retriever.similarity_search(query, top_k=5)

    if not docs:
        return "知识库中未找到相关内容。"

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "未知来源")
        score = doc.metadata.get("score", 0)
        parts.append(f"[文档{i}] 来源: {source} (相似度: {score:.3f})\n{doc.page_content[:800]}")

    return "\n\n---\n\n".join(parts)


# ==================== 工具 2：日志解析 ====================

@tool
def parse_log_content(log_text: str) -> str:
    """解析测试日志，识别 Traceback 和异常类型。

    使用场景：
    - 用户上传了一段测试日志需要分析
    - 需要快速定位日志中的异常信息
    - 需要提取错误的文件和行号

    能识别的异常类型包括：
    TimeoutException, NoSuchElementException, AssertionError,
    ConnectionError, SQLException, OperationalError 等

    Args:
        log_text: 完整的日志文本内容

    Returns:
        解析出的异常列表（JSON 格式），包含异常类型、消息、文件位置等
    """
    from backend.api.deps import get_log_analyzer

    analyzer = get_log_analyzer()
    exceptions = analyzer._extract_exceptions(log_text)

    if not exceptions:
        return "未在日志中检测到已知异常类型。请确认日志内容完整。"

    import json
    exc_list = []
    for exc in exceptions:
        exc_list.append({
            "type": exc.exception_type,
            "message": exc.message[:300],
            "file": exc.file_path,
            "line": exc.line_number,
        })

    summary = f"检测到 {len(exc_list)} 个异常:\n"
    summary += json.dumps(exc_list, ensure_ascii=False, indent=2)
    return summary


# ==================== 工具 3：执行测试 ====================

@tool
def execute_test_scenario(scenario: str) -> str:
    """执行指定的自动化测试场景（沙盒模式）。

    使用场景：
    - 需要验证某个功能是否正常时
    - 需要复现某个缺陷时
    - 需要生成测试报告时

    支持的场景：
    - login: 登录流程测试
    - search: 搜索功能测试
    - order: 下单流程测试

    Args:
        scenario: 测试场景名称 (login / search / order)

    Returns:
        测试执行结果摘要，包含通过/失败状态和详细步骤。
    """
    from backend.models.testing import TestRunRequest, TestScenario

    scenario_map = {
        "login": TestScenario.LOGIN,
        "search": TestScenario.SEARCH,
        "order": TestScenario.ORDER,
    }
    sc = scenario_map.get(scenario.lower())
    if sc is None:
        return f"不支持的场景 '{scenario}'。可用场景: login, search, order"

    from backend.selenium_driver.scenarios.mock_scenarios import (
        run_mock_login_test,
        run_mock_search_test,
        run_mock_order_test,
    )
    runner_map = {
        TestScenario.LOGIN: run_mock_login_test,
        TestScenario.SEARCH: run_mock_search_test,
        TestScenario.ORDER: run_mock_order_test,
    }

    request = TestRunRequest(scenarios=[sc], sandbox=True)
    result = runner_map[sc](request)

    steps_summary = "\n".join(
        f"  {'✓' if s.status == 'passed' else '✗'} {s.step_name} ({s.duration_ms:.0f}ms)"
        + (f" — {s.error_message[:100]}" if s.error_message else "")
        for s in result.steps
    )

    output = f"""场景: {result.scenario}
状态: {result.status}
耗时: {result.duration_ms:.0f}ms
步骤:
{steps_summary}"""

    if result.selenium_logs:
        output += f"\n\n日志摘要:\n{result.selenium_logs[:500]}"

    return output


@tool
def run_real_test_scenario(
    scenario: str,
    base_url: str = "",
    headless: bool = True,
    timeout_seconds: int = 30,
) -> str:
    """在真实 Chrome 浏览器中执行指定的自动化测试场景。

    使用场景：
    - Agent 需要真实验证某个功能是否正常
    - 沙盒模式无法复现问题，需要在真实浏览器中验证
    - 用户要求执行真实测试并分析结果

    支持的场景：
    - login: 登录流程测试
    - search: 搜索功能测试
    - order: 下单流程测试

    Args:
        scenario: 测试场景名称 (login / search / order)
        base_url: 被测网站地址，留空则使用后端内置 Demo 站点
        headless: 是否使用无头模式，默认 True
        timeout_seconds: 超时时间（秒），默认 30

    Returns:
        真实浏览器测试执行结果摘要。
    """
    from backend.api.deps import get_test_executor
    from backend.models.testing import TestRunRequest, TestScenario

    scenario_map = {
        "login": TestScenario.LOGIN,
        "search": TestScenario.SEARCH,
        "order": TestScenario.ORDER,
    }
    sc = scenario_map.get(scenario.lower())
    if sc is None:
        return f"不支持的场景 '{scenario}'。可用场景: login, search, order"

    executor = get_test_executor()
    request = TestRunRequest(
        scenarios=[sc],
        base_url=base_url or "http://localhost:8000/demo",
        headless=headless,
        timeout_seconds=timeout_seconds,
        auto_analyze=False,
        sandbox=True,
    )
    result = executor.run_single_scenario(sc, request)

    steps_summary = "\n".join(
        f"  {'✓' if s.status == 'passed' else '✗'} {s.step_name} ({s.duration_ms:.0f}ms)"
        + (f" — {s.error_message[:100]}" if s.error_message else "")
        for s in result.steps
    )

    output = f"""场景: {result.scenario}
模式: 沙盒模式
状态: {result.status}
耗时: {result.duration_ms:.0f}ms
步骤:
{steps_summary}"""

    if result.error_message:
        output += f"\n\n错误信息:\n{result.error_message[:500]}"
    if result.selenium_logs:
        output += f"\n\n日志摘要:\n{result.selenium_logs[:500]}"

    return output


# ==================== 工具 4：JIRA 缺陷创建 ====================

@tool
def create_jira_issue_tool(title: str, description: str, priority: str = "Medium") -> str:
    """创建 JIRA 缺陷单。

    使用场景：
    - 分析出 bug 后需要提缺陷单
    - 需要将 AI 分析结果同步到 JIRA
    - 需要自动记录测试发现的问题

    Args:
        title: 缺陷标题（简洁明了）
        description: 缺陷描述（含复现步骤、AI 分析等）
        priority: 优先级 (Highest / High / Medium / Low)，默认 Medium

    Returns:
        创建结果，包含 Issue Key 和链接。
    """
    from backend.api.deps import get_jira_creator
    from backend.models.jira import JiraCreateRequest

    request = JiraCreateRequest(
        title=title,
        description=description,
        priority=priority,
        labels=["ai-agent", "react-loop"],
    )
    creator = get_jira_creator()
    response = creator.create_issue(request)

    if response.status == "success":
        return f"缺陷单已创建: {response.issue_key}\n链接: {response.issue_url}\n{response.message}"
    else:
        return f"创建失败: {response.message}"


# ==================== 工具 5：运行日志读取 ====================

@tool
def get_runtime_logs(tail_lines: int = 200, level: str = "all") -> str:
    """读取后端服务最近的运行日志（无需用户上传文件）。

    使用场景：
    - 用户问"刚才什么错误/发生了什么/为什么失败"时，直接拉取运行日志
    - 需要排查后端服务近期异常
    - 需要查看最近的测试执行记录和报错堆栈

    Args:
        tail_lines: 读取最后 N 行日志，默认 200 行（最大 2000）
        level: 日志级别过滤，可选 all / ERROR / WARNING / INFO。
               当用户问"什么错误"时请传 level="ERROR"。

    Returns:
        最近的日志文本（按级别过滤后），含时间戳和模块名。
    """
    from backend.config.settings import get_settings

    settings = get_settings()
    log_dir = Path(settings.get_log_dir())
    log_file = log_dir / "app.log"

    if not log_file.exists():
        return "运行日志文件不存在（服务可能刚启动尚未写入日志）。"

    tail_lines = max(1, min(int(tail_lines), 2000))

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except PermissionError:
        # 文件被其他进程锁定，尝试读取副本
        return "日志文件被占用，无法读取。请稍后重试。"
    except Exception as e:
        return f"读取日志文件失败: {e}"

    if not all_lines:
        return "运行日志为空。"

    # 按级别过滤
    level_upper = level.upper()
    if level_upper in ("ERROR", "WARNING", "INFO"):
        filtered = [ln for ln in all_lines if f"| {level_upper}" in ln]
    else:
        filtered = all_lines

    # 取最后 N 行
    tail = filtered[-tail_lines:]
    return "".join(tail) if tail else f"未找到级别为 {level_upper} 的日志。"


# ==================== 工具 6：系统状态诊断 ====================

@tool
def get_system_status() -> str:
    """诊断后端系统及各依赖组件的运行状态。

    使用场景：
    - 用户问"系统是否正常/能不能用/状态如何"时
    - 排查 LLM、向量库、Chrome 浏览器是否可用
    - 在执行测试前确认环境就绪

    Returns:
        各组件状态报告（LLM / Chroma 向量库 / Chrome 浏览器 / chromedriver）。
    """
    from backend.config.settings import get_settings

    settings = get_settings()
    lines = ["## 系统状态诊断报告"]

    # ---- LLM 配置 ----
    llm_ok = bool(settings.DASHSCOPE_API_KEY)
    lines.append(
        f"- LLM (DashScope): {'✅ 已配置' if llm_ok else '❌ 未配置 DASHSCOPE_API_KEY'}"
        f" | 模型: {settings.LLM_MODEL}"
    )

    # ---- Chroma 向量库 ----
    try:
        from backend.api.deps import get_rag_pipeline
        pipeline = get_rag_pipeline()
        count = pipeline.vector_store.count()
        lines.append(f"- Chroma 向量库: ✅ 正常 | 向量块数: {count}")
    except Exception as e:
        lines.append(f"- Chroma 向量库: ❌ 异常 — {str(e)[:100]}")

    # ---- chromedriver & Chrome 浏览器 ----
    from backend.selenium_driver.driver import detect_chrome
    from backend.selenium_driver.driver import (
        _get_chromedriver_major_version,
        _get_chrome_major_version,
    )

    chrome_binary, driver_path = detect_chrome()
    chrome_version = _get_chrome_major_version() if chrome_binary else None
    driver_version = _get_chromedriver_major_version(driver_path) if driver_path else None

    if not driver_path:
        lines.append("- chromedriver: ❌ 未找到")
    elif chrome_version and driver_version and chrome_version != driver_version:
        lines.append(
            f"- chromedriver: ⚠️ 版本不匹配 | Chrome v{chrome_version} vs chromedriver v{driver_version}"
        )
    elif driver_path:
        lines.append(f"- chromedriver: ✅ {driver_path} (v{driver_version})")

    if not chrome_binary:
        lines.append("- Chrome 浏览器: ❌ 未找到（测试将使用沙盒模式）")
    else:
        lines.append(f"- Chrome 浏览器: ✅ {chrome_binary} (v{chrome_version})")

    return "\n".join(lines)


# ==================== 工具 7：读取代码文件 ====================

@tool
def read_code_file(file_path: str, max_lines: int = 100) -> str:
    """读取项目内的代码文件内容，用于排查问题或理解实现。

    使用场景：
    - 用户问"这段代码为什么出错"时读取相关文件
    - Agent 需要查看后端实现来定位问题
    - 分析代码中的潜在 bug

    Args:
        file_path: 相对于项目根目录的文件路径，例如 backend/agent/graph.py
        max_lines: 最多读取行数，默认 100 行

    Returns:
        文件内容文本，或错误提示。
    """
    from backend.config.settings import PROJECT_ROOT

    target = PROJECT_ROOT / file_path
    # 防止目录遍历
    try:
        target.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return "错误: 只能读取项目根目录内的文件"

    if not target.exists():
        return f"错误: 文件不存在 {target}"
    if not target.is_file():
        return "错误: 路径不是文件"

    try:
        with open(target, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        head = lines[:max_lines]
        content = "".join(head)
        suffix = f"\n\n（共 {total} 行，已显示前 {len(head)} 行）" if total > max_lines else ""
        return content + suffix
    except Exception as e:
        return f"读取文件失败: {e}"


# ==================== 工具 8：列出目录 ====================

@tool
def list_directory(dir_path: str = "") -> str:
    """列出项目内指定目录的内容。

    使用场景：
    - 查找项目结构
    - 确认某个文件是否存在
    - 浏览测试报告目录

    Args:
        dir_path: 相对于项目根目录的目录路径，留空则列出项目根目录

    Returns:
        目录下的文件和子目录列表。
    """
    from backend.config.settings import PROJECT_ROOT

    target = PROJECT_ROOT / dir_path if dir_path else PROJECT_ROOT
    try:
        target.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return "错误: 只能访问项目根目录内的路径"

    if not target.exists():
        return f"错误: 目录不存在 {target}"
    if not target.is_dir():
        return "错误: 路径不是目录"

    items = []
    for p in sorted(target.iterdir()):
        marker = "📁" if p.is_dir() else "📄"
        items.append(f"{marker} {p.name}")
    return "\n".join(items) if items else "（空目录）"


# ==================== 工具 9：执行安全 Shell 命令 ====================

@tool
def run_shell_command(command: str) -> str:
    """执行只允许的白名单 Shell 命令，用于系统诊断。

    使用场景：
    - 查看已安装 Python 包版本
    - 运行 pytest 做快速验证
    - 查看 git 状态（只读类命令）

    Args:
        command: 要执行的命令字符串

    Returns:
        命令输出或错误提示。
    """
    import subprocess
    import shlex
    from backend.config.settings import PROJECT_ROOT

    # 白名单：只允许只读或测试类命令
    allowed_prefixes = (
        "python -m pytest",
        "python -m pip list",
        "python -m pip show",
        "python --version",
        "git status",
        "git log --oneline -n",
        "git diff --stat",
        "dir ",
        "ls ",
        "find ",
    )
    cmd_lower = command.strip().lower()
    if not any(cmd_lower.startswith(p.strip().lower()) for p in allowed_prefixes):
        return (
            f"错误: 命令 '{command}' 不在白名单内。"
            "允许: python -m pytest, python -m pip list/show, git status/log/diff, dir/ls/find"
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output[:4000] or "（命令无输出）"
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时（30秒）"
    except Exception as e:
        return f"命令执行失败: {e}"


# ==================== 工具 10：检查 API 健康 ====================

@tool
def check_api_health(url: str) -> str:
    """检查指定 URL 的接口是否可用。

    使用场景：
    - 检查后端服务是否在线
    - 检查被测系统的接口可用性
    - 排查连接问题

    Args:
        url: 要检查的完整 URL，例如 http://localhost:8000/health

    Returns:
        接口状态、状态码和响应时间。
    """
    import requests
    import time

    try:
        start = time.time()
        resp = requests.get(url, timeout=10)
        elapsed = (time.time() - start) * 1000
        return f"状态: {'可用' if resp.status_code < 400 else '异常'}\n状态码: {resp.status_code}\n响应时间: {elapsed:.1f}ms\n响应内容前 200 字符: {resp.text[:200]}"
    except requests.ConnectionError:
        return f"错误: 无法连接到 {url}"
    except requests.Timeout:
        return f"错误: 请求 {url} 超时"
    except Exception as e:
        return f"检查失败: {e}"


# ==================== 工具 11：获取最近测试记录 ====================

@tool
def get_recent_test_logs(tail_lines: int = 50) -> str:
    """从运行日志中提取最近的测试执行记录。

    使用场景：
    - 查看最近跑了哪些测试
    - 查看测试通过/失败统计
    - 分析测试失败趋势

    Args:
        tail_lines: 读取日志最后多少行，默认 50

    Returns:
        最近的测试相关日志内容。
    """
    from backend.config.settings import get_settings

    settings = get_settings()
    log_file = Path(settings.get_log_dir()) / "app.log"
    if not log_file.exists():
        return "运行日志文件不存在"

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # 过滤包含"测试"关键字的行
        test_lines = [ln for ln in lines if "测试" in ln or "test" in ln.lower()]
        tail = test_lines[-tail_lines:]
        return "".join(tail) if tail else "未找到测试相关日志"
    except Exception as e:
        return f"读取失败: {e}"


# ==================== 工具 12：网站自适应探索 ====================

@tool
def explore_website(url: str, headless: bool = True) -> str:
    """打开指定网站并自动探索页面结构，提取表单、输入框、按钮等信息。

    使用场景：
    - Agent 需要测试一个未知网站时，先探索页面结构
    - 自动识别页面类型（登录页/搜索页/其他）
    - 为后续自定义测试步骤提供定位器信息

    Args:
        url: 要探索的网站 URL，例如 https://www.baidu.com
        headless: 是否使用无头模式，默认 True

    Returns:
        页面结构信息：标题、URL、页面类型、输入框/按钮/链接列表。
    """
    import json
    from backend.selenium_driver.driver import WebDriverManager

    manager = WebDriverManager(headless=headless, timeout_seconds=30)
    try:
        manager.create_driver()
        manager.safe_get(url)

        import time
        time.sleep(2)  # 等待页面动态加载

        page_info = manager.explore_page()

        # 格式化为可读文本
        lines = [
            f"页面标题: {page_info['title']}",
            f"当前 URL: {page_info['url']}",
            f"页面类型: {page_info['page_type']}",
            "",
            f"输入框 ({len(page_info['inputs'])} 个):",
        ]
        for i, inp in enumerate(page_info["inputs"]):
            lines.append(
                f"  [{i+1}] type={inp['type']}, name={inp['name']}, id={inp['id']}, "
                f"placeholder={inp['placeholder'][:30]}, 定位: {inp['by']}={inp['value']}"
            )

        lines.append(f"\n按钮 ({len(page_info['buttons'])} 个):")
        for i, btn in enumerate(page_info["buttons"]):
            lines.append(
                f"  [{i+1}] text={btn['text']}, type={btn['type']}, "
                f"id={btn['id']}, 定位: {btn['by']}={btn['value'][:50]}"
            )

        lines.append(f"\n链接 ({len(page_info['links'])} 个):")
        for i, link in enumerate(page_info["links"][:10]):
            lines.append(f"  [{i+1}] text={link['text']}, href={link['href'][:80]}")

        return "\n".join(lines)

    except Exception as e:
        return f"网站探索失败: {e}"
    finally:
        manager.quit()


# ==================== 工具 13：执行自定义测试场景 ====================

@tool
def run_custom_test(
    url: str,
    steps: str,
    headless: bool = True,
) -> str:
    """在真实浏览器中执行自定义测试步骤。

    使用场景：
    - Agent 探索完网站后，根据页面结构自动生成测试步骤
    - 用户描述测试需求，Agent 转换为步骤并执行

    Args:
        url: 目标网站 URL
        steps: JSON 格式的步骤列表，每个步骤包含:
            - action: navigate/input/click/verify/wait
            - by: id/name/xpath/css_selector (navigate/wait 不需要)
            - value: URL/输入文本(格式: 定位值::输入内容)/定位值/等待秒数
            - description: 步骤描述(可选)
        headless: 是否使用无头模式，默认 True

    Returns:
        测试执行结果摘要。

    示例 steps 参数:
        [
            {"action": "navigate", "value": "https://www.baidu.com"},
            {"action": "input", "by": "id", "value": "kw::Selenium自动化测试"},
            {"action": "click", "by": "id", "value": "su"},
            {"action": "wait", "value": "2"},
            {"action": "verify", "by": "css_selector", "value": "#content_left"}
        ]
    """
    import json
    from backend.api.deps import get_test_executor
    from backend.models.testing import (
        TestRunRequest, TestScenario, CustomScenario,
        CustomTestStep, CustomStepAction,
    )

    try:
        steps_data = json.loads(steps) if isinstance(steps, str) else steps
    except json.JSONDecodeError as e:
        return f"步骤 JSON 解析失败: {e}"

    custom_steps = []
    for s in steps_data:
        custom_steps.append(CustomTestStep(
            action=CustomStepAction(s["action"]),
            by=s.get("by", "id"),
            value=s.get("value", ""),
            description=s.get("description", ""),
        ))

    scenario = CustomScenario(name=f"自定义测试_{url}", steps=custom_steps)
    request = TestRunRequest(
        scenarios=[TestScenario.CUSTOM],
        base_url=url,
        headless=headless,
        timeout_seconds=30,
        auto_analyze=False,
        sandbox=True,
        custom_scenarios=[scenario],
    )

    executor = get_test_executor()
    result = executor.run_custom_scenario(scenario, request)

    steps_summary = "\n".join(
        f"  {'✓' if s.status == 'passed' else '✗'} {s.step_name} ({s.duration_ms:.0f}ms)"
        + (f" — {s.error_message[:100]}" if s.error_message else "")
        for s in result.steps
    )

    return f"""场景: {result.scenario}
状态: {result.status}
耗时: {result.duration_ms:.0f}ms
步骤:
{steps_summary}

{result.selenium_logs[:500] if result.selenium_logs else ''}"""


# ==================== 工具注册表 ====================

# 所有可用工具的列表（供 LangGraph ToolNode 使用）
ALL_TOOLS = [
    search_knowledge_base,
    parse_log_content,
    execute_test_scenario,
    run_real_test_scenario,
    create_jira_issue_tool,
    get_runtime_logs,
    get_system_status,
    read_code_file,
    list_directory,
    run_shell_command,
    check_api_health,
    get_recent_test_logs,
    explore_website,
    run_custom_test,
]

# 按名称索引
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
