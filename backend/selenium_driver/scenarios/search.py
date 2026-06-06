"""
搜索流程自动化测试场景
测试目标：验证 Web 应用的搜索功能是否正常
"""
import time
import logging
from datetime import datetime
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from backend.models.testing import (
    TestRunRequest,
    TestCaseResult,
    TestStepResult,
    TestStatus,
)
from backend.selenium_driver.driver import WebDriverManager

logger = logging.getLogger("ai_rd_agent")


def run_search_test(request: TestRunRequest) -> TestCaseResult:
    """执行搜索流程自动化测试

    测试步骤：
    1. 打开首页
    2. 找到搜索框
    3. 输入搜索关键词
    4. 点击搜索或按回车
    5. 验证搜索结果

    Args:
        request: 测试执行请求

    Returns:
        包含各步骤结果的测试用例结果
    """
    start_time = datetime.now()
    steps = []
    manager = WebDriverManager(
        headless=request.headless,
        timeout_seconds=request.timeout_seconds,
    )
    selenium_logs = []

    try:
        driver = manager.create_driver()
        base_url = request.base_url.rstrip("/")

        # ---- 步骤 1：打开首页 ----
        step_start = time.time()
        try:
            driver.get(base_url)
            selenium_logs.append(f"[INFO] 打开首页: {base_url}")
            steps.append(TestStepResult(
                step_name="打开首页",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
        except TimeoutException as e:
            steps.append(TestStepResult(
                step_name="打开首页",
                status=TestStatus.FAILED,
                error_message=f"页面加载超时: {e}",
            ))
            selenium_logs.append(f"[ERROR] TimeoutException: {e}")
            return _build_search_result(steps, start_time, False, "", selenium_logs)

        # ---- 步骤 2：找到搜索框 ----
        step_start = time.time()
        search_input = None
        search_selectors = [
            ("name", "q"),
            ("name", "search"),
            ("css_selector", "input[type='search']"),
            ("css_selector", "input[placeholder*='搜索'], input[placeholder*='search'], input[placeholder*='Search']"),
            ("xpath", "//input[@aria-label='搜索' or @aria-label='Search']"),
        ]
        for by, selector in search_selectors:
            search_input = manager.safe_find(by, selector, timeout=3)
            if search_input:
                break

        if search_input:
            steps.append(TestStepResult(
                step_name="找到搜索框",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
            selenium_logs.append("[INFO] 搜索框定位成功")
        else:
            steps.append(TestStepResult(
                step_name="找到搜索框",
                status=TestStatus.FAILED,
                error_message="找不到搜索输入框",
            ))
            selenium_logs.append("[ERROR] NoSuchElementException: 找不到搜索框")
            return _build_search_result(steps, start_time, False, "", selenium_logs)

        # ---- 步骤 3：输入搜索关键词 ----
        step_start = time.time()
        try:
            search_input.clear()
            search_input.send_keys("测试关键词")
            selenium_logs.append("[INFO] 输入搜索关键词: 测试关键词")
            steps.append(TestStepResult(
                step_name="输入搜索关键词",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
        except Exception as e:
            steps.append(TestStepResult(
                step_name="输入搜索关键词",
                status=TestStatus.FAILED,
                error_message=f"无法输入搜索词: {e}",
            ))
            selenium_logs.append(f"[ERROR] {type(e).__name__}: {e}")

        # ---- 步骤 4：执行搜索 ----
        step_start = time.time()
        try:
            # 优先查找搜索按钮
            search_btn = (
                manager.safe_find("xpath", "//button[contains(text(), '搜索') or contains(text(), 'Search')]", timeout=3)
                or manager.safe_find("css_selector", "button[type='submit']", timeout=3)
                or manager.safe_find("css_selector", ".search-btn, .search-button", timeout=3)
            )
            if search_btn:
                search_btn.click()
                selenium_logs.append("[INFO] 点击搜索按钮")
            else:
                # 备选：按回车触发搜索
                search_input.send_keys("\n")
                selenium_logs.append("[INFO] 按回车触发搜索")

            steps.append(TestStepResult(
                step_name="执行搜索",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
        except Exception as e:
            steps.append(TestStepResult(
                step_name="执行搜索",
                status=TestStatus.FAILED,
                error_message=f"搜索操作失败: {e}",
            ))
            selenium_logs.append(f"[ERROR] {type(e).__name__}: {e}")

        # ---- 步骤 5：验证搜索结果 ----
        step_start = time.time()
        time.sleep(1.5)
        result_found = False
        result_selectors = [
            ("css_selector", ".search-results, .results, .search-result-list"),
            ("xpath", "//*[contains(@class, 'result') or contains(@class, 'search')]"),
            ("css_selector", "[data-testid='search-results']"),
        ]
        for by, selector in result_selectors:
            if manager.safe_find(by, selector, timeout=2):
                result_found = True
                break

        if result_found:
            steps.append(TestStepResult(
                step_name="验证搜索结果",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
            selenium_logs.append("[INFO] 搜索结果验证通过")
        else:
            steps.append(TestStepResult(
                step_name="验证搜索结果",
                status=TestStatus.FAILED,
                error_message="搜索后未显示结果区域",
            ))
            selenium_logs.append("[WARN] 未检测到搜索结果，可能需要调整验证逻辑")

        all_passed = all(s.status == TestStatus.PASSED for s in steps)
        return _build_search_result(
            steps, start_time, all_passed,
            "" if all_passed else "搜索流程存在问题",
            selenium_logs,
        )

    except Exception as e:
        logger.error(f"搜索测试异常: {e}", exc_info=True)
        selenium_logs.append(f"[FATAL] {type(e).__name__}: {e}")
        return _build_search_result(
            steps, start_time, False, str(e), selenium_logs,
        )
    finally:
        manager.quit()


def _build_search_result(
    steps: list[TestStepResult],
    start_time: datetime,
    all_passed: bool,
    error_message: str,
    selenium_logs: list[str],
) -> TestCaseResult:
    """构建搜索测试结果"""
    end_time = datetime.now()
    duration_ms = (end_time - start_time).total_seconds() * 1000

    return TestCaseResult(
        scenario="search",
        status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        steps=steps,
        error_message=error_message,
        selenium_logs="\n".join(selenium_logs),
    )
