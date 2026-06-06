"""
Mock 测试场景模块 — 沙盒模式
模拟 Selenium 执行过程，无需 Chrome 浏览器即可演示完整的测试→失败→分析流程

每个 mock 函数模拟真实测试场景的行为，包括：
- 模拟的步骤耗时（用 time.sleep 模拟）
- 随机成功/失败（支持配置失败率）
- 真实的错误日志格式
"""
import time
import random
import logging
from datetime import datetime

from backend.models.testing import (
    TestRunRequest,
    TestCaseResult,
    TestStepResult,
    TestStatus,
)

logger = logging.getLogger("ai_rd_agent")

# 模拟失败率（用于演示，默认 30% 场景会失败）
MOCK_FAILURE_RATE = 0.3


def run_mock_login_test(request: TestRunRequest) -> TestCaseResult:
    """模拟登录流程测试（沙盒模式）

    模拟步骤：打开页面 → 输入用户名 → 输入密码 → 点击登录 → 验证结果
    """
    start_time = datetime.now()
    steps = []
    selenium_logs = []
    rng = random.Random()

    # 随机决定这个场景是否失败
    will_fail = rng.random() < MOCK_FAILURE_RATE

    try:
        # 步骤 1：打开登录页面
        time.sleep(0.1)
        steps.append(TestStepResult(
            step_name="打开登录页面",
            status=TestStatus.PASSED,
            duration_ms=rng.uniform(800, 1500),
        ))
        selenium_logs.append(f"[MOCK] 打开 {request.base_url}/login")

        # 步骤 2：输入用户名
        time.sleep(0.05)
        steps.append(TestStepResult(
            step_name="输入用户名",
            status=TestStatus.PASSED,
            duration_ms=rng.uniform(100, 300),
        ))
        selenium_logs.append("[MOCK] 输入用户名: test_user@example.com")

        # 步骤 3：输入密码
        time.sleep(0.05)
        steps.append(TestStepResult(
            step_name="输入密码",
            status=TestStatus.PASSED,
            duration_ms=rng.uniform(100, 300),
        ))
        selenium_logs.append("[MOCK] 输入密码: ********")

        # 步骤 4：点击登录按钮
        time.sleep(0.1)
        steps.append(TestStepResult(
            step_name="点击登录按钮",
            status=TestStatus.PASSED,
            duration_ms=rng.uniform(200, 600),
        ))
        selenium_logs.append("[MOCK] 点击登录按钮")

        # 步骤 5：验证登录结果 — 如果 will_fail，模拟超时
        time.sleep(0.1)
        if will_fail:
            error_msg = (
                "TimeoutException: Page load timeout after 30000ms\n"
                "  File 'test_login.py', line 45, in test_login\n"
                "    driver.get('https://example.com/login')\n"
                "  File 'selenium/webdriver/remote/webdriver.py', line 333\n"
                "    self.execute(Command.GET, {'url': url})\n"
                "AssertionError: Login test failed - expected dashboard but got timeout"
            )
            steps.append(TestStepResult(
                step_name="验证登录成功",
                status=TestStatus.FAILED,
                duration_ms=rng.uniform(1000, 3000),
                error_message=error_msg[:500],
            ))
            selenium_logs.append(f"[MOCK ERROR] {error_msg}")
        else:
            steps.append(TestStepResult(
                step_name="验证登录成功",
                status=TestStatus.PASSED,
                duration_ms=rng.uniform(200, 500),
            ))
            selenium_logs.append("[MOCK] 登录成功，已跳转到 Dashboard")

    except Exception as e:
        selenium_logs.append(f"[MOCK FATAL] {e}")
        steps.append(TestStepResult(
            step_name="测试执行",
            status=TestStatus.ERROR,
            error_message=str(e),
        ))

    all_passed = all(s.status == TestStatus.PASSED for s in steps)
    end_time = datetime.now()
    duration_ms = (end_time - start_time).total_seconds() * 1000

    return TestCaseResult(
        scenario="login",
        status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        steps=steps,
        error_message="" if all_passed else "模拟登录超时 — 用于演示故障分析流程",
        selenium_logs="\n".join(selenium_logs),
    )


def run_mock_search_test(request: TestRunRequest) -> TestCaseResult:
    """模拟搜索流程测试（沙盒模式）

    模拟步骤：打开首页 → 定位搜索框 → 输入关键词 → 执行搜索 → 验证结果
    """
    start_time = datetime.now()
    steps = []
    selenium_logs = []
    rng = random.Random()

    will_fail = rng.random() < MOCK_FAILURE_RATE

    try:
        # 步骤 1：打开首页
        time.sleep(0.1)
        steps.append(TestStepResult(
            step_name="打开首页", status=TestStatus.PASSED,
            duration_ms=rng.uniform(500, 1200),
        ))
        selenium_logs.append(f"[MOCK] 打开 {request.base_url}")

        # 步骤 2：定位搜索框
        time.sleep(0.05)
        steps.append(TestStepResult(
            step_name="找到搜索框", status=TestStatus.PASSED,
            duration_ms=rng.uniform(50, 200),
        ))
        selenium_logs.append("[MOCK] 搜索框定位成功: input[name='q']")

        # 步骤 3：输入关键词
        time.sleep(0.05)
        steps.append(TestStepResult(
            step_name="输入搜索关键词", status=TestStatus.PASSED,
            duration_ms=rng.uniform(100, 300),
        ))
        selenium_logs.append("[MOCK] 输入关键词: 测试用例")

        # 步骤 4：执行搜索
        time.sleep(0.1)
        steps.append(TestStepResult(
            step_name="执行搜索", status=TestStatus.PASSED,
            duration_ms=rng.uniform(200, 500),
        ))
        selenium_logs.append("[MOCK] 点击搜索按钮")

        # 步骤 5：验证结果
        time.sleep(0.1)
        if will_fail:
            error_msg = (
                "NoSuchElementException: Unable to locate element: "
                "{'method':'css selector','selector':'.search-results'}\n"
                "  File 'test_search.py', line 78, in verify_search_results\n"
                "    results = driver.find_element(By.CSS_SELECTOR, '.search-results')"
            )
            steps.append(TestStepResult(
                step_name="验证搜索结果",
                status=TestStatus.FAILED,
                duration_ms=rng.uniform(500, 1500),
                error_message=error_msg[:500],
            ))
            selenium_logs.append(f"[MOCK ERROR] {error_msg}")
        else:
            steps.append(TestStepResult(
                step_name="验证搜索结果",
                status=TestStatus.PASSED,
                duration_ms=rng.uniform(100, 400),
            ))
            selenium_logs.append("[MOCK] 搜索结果显示正常，共 42 条结果")

    except Exception as e:
        selenium_logs.append(f"[MOCK FATAL] {e}")

    all_passed = all(s.status == TestStatus.PASSED for s in steps)
    end_time = datetime.now()
    duration_ms = (end_time - start_time).total_seconds() * 1000

    return TestCaseResult(
        scenario="search",
        status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        steps=steps,
        error_message="" if all_passed else "模拟元素定位失败 — 用于演示故障分析流程",
        selenium_logs="\n".join(selenium_logs),
    )


def run_mock_order_test(request: TestRunRequest) -> TestCaseResult:
    """模拟下单流程测试（沙盒模式）

    模拟步骤：打开商品页 → 点击下单 → 填写表单 → 提交订单 → 验证结果
    """
    start_time = datetime.now()
    steps = []
    selenium_logs = []
    rng = random.Random()

    will_fail = rng.random() < MOCK_FAILURE_RATE

    try:
        # 步骤 1：打开商品页面
        time.sleep(0.1)
        steps.append(TestStepResult(
            step_name="打开目标页面", status=TestStatus.PASSED,
            duration_ms=rng.uniform(600, 1400),
        ))
        selenium_logs.append(f"[MOCK] 打开 {request.base_url}/product/1")

        # 步骤 2：点击下单按钮
        time.sleep(0.05)
        steps.append(TestStepResult(
            step_name="找到操作按钮", status=TestStatus.PASSED,
            duration_ms=rng.uniform(100, 300),
        ))
        selenium_logs.append("[MOCK] 找到下单按钮")

        # 步骤 3：填写表单
        time.sleep(0.1)
        steps.append(TestStepResult(
            step_name="填写表单字段", status=TestStatus.PASSED,
            duration_ms=rng.uniform(300, 800),
        ))
        selenium_logs.append("[MOCK] 填写收货地址、联系方式等")

        # 步骤 4：提交订单
        time.sleep(0.1)
        steps.append(TestStepResult(
            step_name="提交订单/表单", status=TestStatus.PASSED,
            duration_ms=rng.uniform(500, 1200),
        ))
        selenium_logs.append("[MOCK] 点击提交订单")

        # 步骤 5：验证结果
        if will_fail:
            error_msg = (
                "AssertionError: Expected order total 299.00 but got 329.00\n"
                "  File 'test_order.py', line 156, in verify_order_total\n"
                "    assert actual_total == expected_total, f'Expected {expected_total} but got {actual_total}'"
            )
            steps.append(TestStepResult(
                step_name="验证提交结果",
                status=TestStatus.FAILED,
                duration_ms=rng.uniform(500, 2000),
                error_message=error_msg[:500],
            ))
            selenium_logs.append(f"[MOCK ERROR] {error_msg}")
        else:
            steps.append(TestStepResult(
                step_name="验证提交结果",
                status=TestStatus.PASSED,
                duration_ms=rng.uniform(200, 600),
            ))
            selenium_logs.append("[MOCK] 订单提交成功，订单号: ORD-2024-001234")

    except Exception as e:
        selenium_logs.append(f"[MOCK FATAL] {e}")

    all_passed = all(s.status == TestStatus.PASSED for s in steps)
    end_time = datetime.now()
    duration_ms = (end_time - start_time).total_seconds() * 1000

    return TestCaseResult(
        scenario="order",
        status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        steps=steps,
        error_message="" if all_passed else "模拟金额校验失败 — 用于演示故障分析流程",
        selenium_logs="\n".join(selenium_logs),
    )
