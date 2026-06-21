"""
登录流程自动化测试场景
测试目标：验证 Web 应用的登录功能是否正常
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


def run_login_test(request: TestRunRequest) -> TestCaseResult:
    """执行登录流程自动化测试

    测试步骤：
    1. 打开登录页面
    2. 输入用户名
    3. 输入密码
    4. 点击登录按钮
    5. 验证登录成功（检查页面跳转或欢迎元素）

    Args:
        request: 测试执行请求（含 base_url、headless、timeout 等配置）

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

        # ---- 步骤 1：打开登录页面 ----
        step_start = time.time()
        try:
            driver.get(f"{base_url}/login")
            selenium_logs.append("[INFO] 打开登录页面")
            steps.append(TestStepResult(
                step_name="打开登录页面",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
        except TimeoutException as e:
            steps.append(TestStepResult(
                step_name="打开登录页面",
                status=TestStatus.FAILED,
                error_message=f"页面加载超时: {e}",
            ))
            selenium_logs.append(f"[ERROR] TimeoutException: {e}")
            return _build_result("login", steps, start_time, False, "", selenium_logs)

        # ---- 步骤 2：输入用户名 ----
        step_start = time.time()
        try:
            username_input = manager.wait_for_element("name", "username", timeout=10)
            username_input.clear()
            username_input.send_keys("test_user@example.com")
            selenium_logs.append("[INFO] 输入用户名: test_user@example.com")
            steps.append(TestStepResult(
                step_name="输入用户名",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
        except (TimeoutException, NoSuchElementException) as e:
            # 尝试备选定位方式
            alt_input = manager.safe_find("css_selector", "input[type='email']", timeout=3)
            if alt_input:
                alt_input.send_keys("test_user@example.com")
                steps.append(TestStepResult(step_name="输入用户名", status=TestStatus.PASSED))
                selenium_logs.append("[INFO] 使用备选定位器输入用户名")
            else:
                steps.append(TestStepResult(
                    step_name="输入用户名",
                    status=TestStatus.FAILED,
                    error_message=f"找不到用户名输入框: {e}",
                ))
                selenium_logs.append(f"[ERROR] NoSuchElementException: {e}")

        # ---- 步骤 3：输入密码 ----
        step_start = time.time()
        try:
            password_input = manager.wait_for_element("name", "password", timeout=10)
            password_input.clear()
            password_input.send_keys("test_password_123")
            selenium_logs.append("[INFO] 输入密码")
            steps.append(TestStepResult(
                step_name="输入密码",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
        except (TimeoutException, NoSuchElementException) as e:
            alt_input = manager.safe_find("css_selector", "input[type='password']", timeout=3)
            if alt_input:
                alt_input.send_keys("test_password_123")
                steps.append(TestStepResult(step_name="输入密码", status=TestStatus.PASSED))
                selenium_logs.append("[INFO] 使用备选定位器输入密码")
            else:
                steps.append(TestStepResult(
                    step_name="输入密码",
                    status=TestStatus.FAILED,
                    error_message=f"找不到密码输入框: {e}",
                ))
                selenium_logs.append(f"[ERROR] NoSuchElementException: {e}")

        # ---- 步骤 4：点击登录按钮 ----
        step_start = time.time()
        try:
            login_btn = manager.wait_for_clickable(
                "xpath",
                "//button[contains(text(), '登录') or contains(text(), 'Login') or @type='submit']",
                timeout=10,
            )
            login_btn.click()
            selenium_logs.append("[INFO] 点击登录按钮")
            steps.append(TestStepResult(
                step_name="点击登录按钮",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
        except TimeoutException as e:
            # 备选：通过 id 或 class 查找按钮
            alt_btn = (
                manager.safe_find("id", "login-btn", timeout=3)
                or manager.safe_find("css_selector", ".login-button", timeout=3)
            )
            if alt_btn:
                alt_btn.click()
                steps.append(TestStepResult(step_name="点击登录按钮", status=TestStatus.PASSED))
                selenium_logs.append("[INFO] 使用备选定位器点击登录按钮")
            else:
                steps.append(TestStepResult(
                    step_name="点击登录按钮",
                    status=TestStatus.FAILED,
                    error_message=f"找不到登录按钮: {e}",
                ))
                selenium_logs.append(f"[ERROR] NoSuchElementException: 找不到登录按钮")

        # ---- 步骤 5：验证登录成功 ----
        step_start = time.time()
        time.sleep(2)  # 等待页面跳转
        success = False
        success_indicators = [
            ("xpath", "//*[contains(text(), '欢迎') or contains(text(), 'Welcome')]"),
            ("css_selector", ".user-menu, .user-profile, .dashboard"),
            ("css_selector", "[data-testid='user-menu']"),
        ]
        for by, selector in success_indicators:
            if manager.safe_find(by, selector, timeout=3):
                success = True
                break

        if success:
            steps.append(TestStepResult(
                step_name="验证登录成功",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
            selenium_logs.append("[INFO] 登录成功验证通过")
        else:
            steps.append(TestStepResult(
                step_name="验证登录成功",
                status=TestStatus.FAILED,
                error_message="未检测到登录成功的标志元素",
            ))
            selenium_logs.append("[WARN] 未检测到登录成功标志，可能需要调整验证逻辑")

        # 判断整体结果
        all_passed = all(s.status == TestStatus.PASSED for s in steps)
        return _build_result(
            "login", steps, start_time, all_passed,
            "" if all_passed else "部分步骤执行失败",
            selenium_logs,
        )

    except Exception as e:
        logger.error(f"登录测试异常: {e}", exc_info=True)
        selenium_logs.append(f"[FATAL] {type(e).__name__}: {e}")
        steps.append(TestStepResult(
            step_name="测试执行",
            status=TestStatus.ERROR,
            error_message=str(e),
        ))
        return _build_result("login", steps, start_time, False, str(e), selenium_logs)
    finally:
        manager.quit()


def _build_result(
    scenario: str,
    steps: list[TestStepResult],
    start_time: datetime,
    all_passed: bool,
    error_message: str,
    selenium_logs: list[str],
) -> TestCaseResult:
    """构建测试用例结果"""
    end_time = datetime.now()
    duration_ms = (end_time - start_time).total_seconds() * 1000

    return TestCaseResult(
        scenario=scenario,
        status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        steps=steps,
        error_message=error_message,
        selenium_logs="\n".join(selenium_logs),
    )
