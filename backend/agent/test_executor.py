"""
测试执行 Agent 模块
负责调度 Selenium 测试场景、收集结果，并在失败时触发日志分析
"""
import logging
import uuid
from datetime import datetime

from backend.models.testing import (
    TestRunRequest,
    TestReport,
    TestCaseResult,
    TestStatus,
    TestScenario,
)
from backend.agent.log_analyzer import LogAnalyzer
from backend.models.analysis import LogAnalysisRequest

logger = logging.getLogger("ai_rd_agent")


class TestExecutorAgent:
    """测试执行 Agent

    职责：
    1. 接收测试请求，调度各业务场景的执行
    2. 收集每个场景的执行结果
    3. 对失败用例自动调用日志分析 Agent
    4. 汇总生成完整的测试报告

    使用示例：
        executor = TestExecutorAgent()
        report = executor.run_tests(TestRunRequest(
            scenarios=[TestScenario.LOGIN, TestScenario.SEARCH],
            base_url="https://example.com",
        ))
    """

    def __init__(self, log_analyzer: LogAnalyzer | None = None):
        if log_analyzer:
            self.log_analyzer = log_analyzer
        else:
            from backend.api.deps import get_log_analyzer
            self.log_analyzer = get_log_analyzer()
        logger.info("测试执行 Agent 已初始化")

    def run_tests(self, request: TestRunRequest) -> TestReport:
        """执行整个测试套件

        Args:
            request: 测试执行请求（场景、目标 URL、配置参数）

        Returns:
            完整的测试报告
        """
        report_id = str(uuid.uuid4())[:8]
        logger.info(f"开始执行测试套件 [{report_id}]: {len(request.scenarios)} 个场景")

        results = []
        for scenario in request.scenarios:
            result = self._run_single_scenario(scenario, request)
            results.append(result)

        # 统计
        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)

        # 对失败用例进行分析
        failure_analysis = []
        if request.auto_analyze:
            for result in results:
                if result.status == TestStatus.FAILED and result.selenium_logs:
                    analysis = self._analyze_failure(result)
                    if analysis:
                        failure_analysis.append(analysis)

        report = TestReport(
            report_id=report_id,
            base_url=request.base_url,
            total_scenarios=total,
            passed_count=passed,
            failed_count=failed,
            pass_rate=passed / total if total > 0 else 0.0,
            results=results,
            failure_analysis=failure_analysis,
        )

        logger.info(
            f"测试套件完成 [{report_id}]: "
            f"通过 {passed}/{total} ({report.pass_rate:.0%})"
        )
        return report

    def _run_single_scenario(
        self,
        scenario: TestScenario,
        request: TestRunRequest,
    ) -> TestCaseResult:
        """执行单个测试场景

        优先使用真实 Selenium；当 sandbox=True 或 Chrome 不可用时使用 mock 模式。

        Args:
            scenario: 测试场景枚举
            request: 测试请求参数

        Returns:
            用例执行结果
        """
        logger.info(f"  执行场景: {scenario.value} (sandbox={request.sandbox})")

        # 判断是否使用沙盒模式
        use_mock = request.sandbox or not self._chrome_available()

        try:
            if use_mock:
                # ---- 沙盒模式：模拟 Selenium 执行 ----
                from backend.selenium_driver.scenarios.mock_scenarios import (
                    run_mock_login_test,
                    run_mock_search_test,
                    run_mock_order_test,
                )
                scenario_map = {
                    TestScenario.LOGIN: run_mock_login_test,
                    TestScenario.SEARCH: run_mock_search_test,
                    TestScenario.ORDER: run_mock_order_test,
                }
            else:
                # ---- 真实 Selenium 模式 ----
                from backend.selenium_driver.scenarios.login import run_login_test
                from backend.selenium_driver.scenarios.search import run_search_test
                from backend.selenium_driver.scenarios.order import run_order_test
                scenario_map = {
                    TestScenario.LOGIN: run_login_test,
                    TestScenario.SEARCH: run_search_test,
                    TestScenario.ORDER: run_order_test,
                }

            runner = scenario_map.get(scenario)
            if runner is None:
                return TestCaseResult(
                    scenario=scenario.value,
                    status=TestStatus.SKIPPED,
                    error_message=f"未知场景: {scenario.value}",
                )

            # 调用场景执行函数（mock 或真实 Selenium）
            result = runner(request)
            logger.info(
                f"  场景 {scenario.value} 完成: {result.status.value} "
                f"({result.duration_ms:.0f}ms){' [MOCK]' if use_mock else ''}"
            )
            return result

        except ImportError as e:
            logger.warning(f"场景模块导入失败: {e}")
            return TestCaseResult(
                scenario=scenario.value,
                status=TestStatus.SKIPPED,
                error_message=f"场景模块导入失败: {e}",
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
        except Exception as e:
            logger.error(f"场景执行异常 {scenario.value}: {e}", exc_info=True)
            return TestCaseResult(
                scenario=scenario.value,
                status=TestStatus.ERROR,
                error_message=str(e),
            )

    @staticmethod
    def _chrome_available() -> bool:
        """检测 Chrome 浏览器和 chromedriver 是否可用（轻量检测，不启动浏览器）"""
        from backend.selenium_driver.driver import detect_chrome

        chrome_binary, driver_path = detect_chrome()
        if not driver_path:
            logger.info("Chrome 不可用：未找到 chromedriver，将使用沙盒模式")
            return False
        if not chrome_binary:
            logger.info("Chrome 不可用：未找到 Chrome 浏览器，将使用沙盒模式")
            return False

        logger.info(f"Chrome 检测通过: browser={chrome_binary}, driver={driver_path}")
        return True

    def _analyze_failure(self, result: TestCaseResult) -> str:
        """对失败用例进行 AI 分析

        Args:
            result: 失败的测试用例结果

        Returns:
            AI 分析结果的文本摘要
        """
        try:
            log_content = result.error_message or ""
            if result.selenium_logs:
                log_content += f"\n\n[Selenium Logs]\n{result.selenium_logs}"

            if not log_content.strip():
                return ""

            analysis_request = LogAnalysisRequest(
                log_content=log_content[:5000],
                filename=f"{result.scenario}_test.log",
                include_historical=True,
            )
            analysis_result = self.log_analyzer.analyze(analysis_request)

            return (
                f"[{result.scenario}] {analysis_result.summary[:200]}"
            )
        except Exception as e:
            logger.warning(f"失败分析异常: {e}")
            return f"[{result.scenario}] 分析失败: {str(e)}"
