"""
Agent 工具定义模块
使用 LangChain @tool 装饰器，将后端能力封装为 Agent 可调用的工具函数
"""
import logging
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
    from backend.rag.retriever import Retriever

    retriever = Retriever()
    docs = retriever.similarity_search(query, top_k=5)

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
    from backend.agent.log_analyzer import LogAnalyzer

    analyzer = LogAnalyzer()
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
    from backend.models.jira import JiraCreateRequest
    from backend.agent.jira_creator import JiraCreator

    request = JiraCreateRequest(
        title=title,
        description=description,
        priority=priority,
        labels=["ai-agent", "react-loop"],
    )
    creator = JiraCreator()
    response = creator.create_issue(request)

    if response.status == "success":
        return f"缺陷单已创建: {response.issue_key}\n链接: {response.issue_url}\n{response.message}"
    else:
        return f"创建失败: {response.message}"


# ==================== 工具注册表 ====================

# 所有可用工具的列表（供 LangGraph ToolNode 使用）
ALL_TOOLS = [
    search_knowledge_base,
    parse_log_content,
    execute_test_scenario,
    create_jira_issue_tool,
]

# 按名称索引
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
