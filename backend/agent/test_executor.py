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
    TestStepResult,
    TestStatus,
    TestScenario,
    CustomScenario,
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

    def run_tests(
        self,
        request: TestRunRequest,
        cancel_event=None,
    ) -> TestReport:
        """执行整个测试套件

        Args:
            request: 测试执行请求（场景、目标 URL、配置参数）
            cancel_event: 可选的线程事件，用于接收取消信号

        Returns:
            完整的测试报告
        """
        report_id = str(uuid.uuid4())[:8]
        logger.info(f"开始执行测试套件 [{report_id}]: {len(request.scenarios)} 个场景")

        results = []
        use_mock = request.sandbox or not self._chrome_available()

        # 真实浏览器模式下创建 SessionManager（同批次共享一个 WebDriver）
        session_manager = None
        if not use_mock:
            from backend.selenium_driver.session_manager import SessionManager
            try:
                session_manager = SessionManager(
                    headless=request.headless,
                    timeout_seconds=request.timeout_seconds,
                )
                session_manager.create_driver()
            except Exception as e:
                logger.warning(f"共享浏览器创建失败，回退到各场景独立实例: {e}")
                session_manager = None

        try:
            # 执行预定义场景
            for scenario in request.scenarios:
                if cancel_event and cancel_event.is_set():
                    logger.info(f"测试套件 [{report_id}] 收到取消信号，停止执行")
                    break
                if scenario == TestScenario.CUSTOM:
                    continue
                if session_manager is not None:
                    session_manager.isolate(request.base_url)
                result = self.run_single_scenario(scenario, request, use_mock, session_manager)
                results.append(result)

            # 执行自定义场景
            for custom_scn in request.custom_scenarios:
                if cancel_event and cancel_event.is_set():
                    logger.info(f"测试套件 [{report_id}] 收到取消信号，停止执行")
                    break
                if session_manager is not None:
                    session_manager.isolate(request.base_url)
                result = self.run_custom_scenario(custom_scn, request, session_manager)
                results.append(result)
        finally:
            if session_manager is not None:
                try:
                    session_manager.end_session()
                except Exception:
                    pass

        # 将未执行的场景标记为 cancelled
        all_scenario_names = [
            s.value for s in request.scenarios if s != TestScenario.CUSTOM
        ] + [cs.name for cs in request.custom_scenarios]
        executed_names = {r.scenario for r in results}
        for name in all_scenario_names:
            if name not in executed_names:
                results.append(TestCaseResult(
                    scenario=name,
                    status=TestStatus.CANCELLED,
                    error_message="任务已取消",
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                ))

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

    def run_single_scenario(
        self,
        scenario: TestScenario,
        request: TestRunRequest,
        use_mock: bool = False,
        shared_manager=None,
    ) -> TestCaseResult:
        """执行单个测试场景

        Args:
            scenario: 测试场景枚举
            request: 测试请求参数
            use_mock: 是否使用沙盒模式
            shared_manager: 共享浏览器实例

        Returns:
            用例执行结果
        """
        logger.info(f"  执行场景: {scenario.value} (sandbox={request.sandbox})")

        try:
            if use_mock:
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

            # 传入共享浏览器实例
            result = runner(request, shared_manager)
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

    def run_custom_scenario(
        self,
        scenario: "CustomScenario",
        request: TestRunRequest,
        shared_manager=None,
    ) -> TestCaseResult:
        """执行自定义测试场景

        Args:
            scenario: 自定义场景定义
            request: 测试请求参数
            shared_manager: 共享浏览器实例

        Returns:
            用例执行结果
        """
        from backend.models.testing import CustomScenario  # noqa: F811

        logger.info(f"  执行自定义场景: {scenario.name} ({len(scenario.steps)} 步)")

        if request.sandbox or not self._chrome_available():
            # 沙盒模式：模拟执行
            logger.info(f"  自定义场景 {scenario.name} 使用沙盒模式")
            return self._mock_custom_scenario(scenario)

        try:
            from backend.selenium_driver.scenarios.custom import run_custom_scenario
            result = run_custom_scenario(request, scenario, shared_manager)
            logger.info(
                f"  自定义场景 {scenario.name} 完成: {result.status.value} "
                f"({result.duration_ms:.0f}ms)"
            )
            return result
        except Exception as e:
            logger.error(f"自定义场景执行异常 {scenario.name}: {e}", exc_info=True)
            return TestCaseResult(
                scenario=scenario.name,
                status=TestStatus.ERROR,
                error_message=str(e),
            )

    @staticmethod
    def _mock_custom_scenario(scenario: "CustomScenario") -> TestCaseResult:
        """沙盒模式下模拟执行自定义场景"""
        import time as _time
        from datetime import datetime

        start = datetime.now()
        steps = []
        for i, s in enumerate(scenario.steps):
            _time.sleep(0.1)
            steps.append(TestStepResult(
                step_name=s.description or f"步骤{i+1}: {s.action.value}",
                status=TestStatus.PASSED,
                duration_ms=100,
            ))
        all_passed = all(s.status == TestStatus.PASSED for s in steps) if steps else False
        return TestCaseResult(
            scenario=scenario.name,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            start_time=start,
            end_time=datetime.now(),
            duration_ms=100 * len(steps),
            steps=steps,
            selenium_logs="[MOCK] 沙盒模式模拟执行",
        )

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
