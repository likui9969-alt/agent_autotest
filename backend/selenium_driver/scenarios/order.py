"""
下单流程自动化测试场景
测试目标：验证 Web 应用的下单/提交表单功能是否正常
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


def run_order_test(request: TestRunRequest) -> TestCaseResult:
    """执行下单流程自动化测试

    测试步骤：
    1. 打开目标页面（如商品详情/表单页）
    2. 查找并点击"下单"/"提交"/"购买"按钮
    3. 填写必填表单字段
    4. 提交订单/表单
    5. 验证提交结果

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

        # ---- 步骤 1：打开目标页面 ----
        step_start = time.time()
        target_urls = [
            f"{base_url}/order",
            f"{base_url}/checkout",
            f"{base_url}/product/1",
            base_url,
        ]
        page_loaded = False
        for url in target_urls:
            try:
                manager.safe_get(url)
                selenium_logs.append(f"[INFO] 打开页面: {url}")
                page_loaded = True
                break
            except TimeoutException:
                selenium_logs.append(f"[WARN] {url} 加载超时，尝试下一个...")
                continue

        if page_loaded:
            steps.append(TestStepResult(
                step_name="打开目标页面",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
        else:
            steps.append(TestStepResult(
                step_name="打开目标页面",
                status=TestStatus.FAILED,
                error_message="所有目标 URL 均加载失败",
            ))
            return _build_order_result(steps, start_time, False, "", selenium_logs)

        # ---- 步骤 2：查找下单/提交按钮 ----
        step_start = time.time()
        action_btn = None
        btn_selectors = [
            ("xpath", "//button[contains(text(), '下单') or contains(text(), '购买') or contains(text(), '提交') or contains(text(), 'Order') or contains(text(), 'Buy') or contains(text(), 'Submit')]"),
            ("css_selector", ".order-btn, .buy-btn, .submit-btn, .checkout-btn"),
            ("xpath", "//a[contains(text(), '下单') or contains(text(), '购买')]"),
        ]
        for by, selector in btn_selectors:
            action_btn = manager.safe_find(by, selector, timeout=3)
            if action_btn:
                break

        if action_btn:
            steps.append(TestStepResult(
                step_name="找到操作按钮",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
            selenium_logs.append("[INFO] 操作按钮定位成功")
        else:
            steps.append(TestStepResult(
                step_name="找到操作按钮",
                status=TestStatus.FAILED,
                error_message="找不到下单/提交/购买按钮",
            ))
            selenium_logs.append("[ERROR] 找不到操作按钮")

        # ---- 步骤 3：填写表单字段（如有） ----
        step_start = time.time()
        text_inputs_filled = 0
        form_selectors = [
            ("name", "name"),
            ("name", "address"),
            ("name", "phone"),
            ("name", "email"),
            ("css_selector", "input[required]"),
        ]
        test_data = {
            "name": "测试用户",
            "address": "测试地址 123 号",
            "phone": "13800138000",
            "email": "test@example.com",
        }

        for by, selector in form_selectors:
            field = manager.safe_find(by, selector, timeout=2)
            if field:
                field_name = selector.replace("name", "").replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                try:
                    # 获取字段名，尝试填对应的测试数据
                    actual_name = field.get_attribute("name") or field.get_attribute("placeholder") or field_name
                    fill_value = test_data.get(actual_name, "测试数据")
                    field.clear()
                    field.send_keys(fill_value)
                    text_inputs_filled += 1
                except Exception as e:
                    selenium_logs.append(f"[WARN] 填写字段 {field_name} 失败: {e}")

        selenium_logs.append(f"[INFO] 填写了 {text_inputs_filled} 个表单字段")
        steps.append(TestStepResult(
            step_name="填写表单字段",
            status=TestStatus.PASSED,
            duration_ms=(time.time() - step_start) * 1000,
        ))

        # ---- 步骤 4：提交订单/表单 ----
        step_start = time.time()
        try:
            if action_btn:
                action_btn.click()
                selenium_logs.append("[INFO] 点击提交按钮")
            else:
                # 尝试提交表单的通用方式
                form = manager.safe_find("tag name", "form", timeout=2)
                if form:
                    form.submit()
                    selenium_logs.append("[INFO] 通过 form.submit() 提交")

            steps.append(TestStepResult(
                step_name="提交订单/表单",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
        except Exception as e:
            steps.append(TestStepResult(
                step_name="提交订单/表单",
                status=TestStatus.FAILED,
                error_message=f"提交失败: {e}",
            ))
            selenium_logs.append(f"[ERROR] {type(e).__name__}: {e}")

        # ---- 步骤 5：验证提交结果 ----
        step_start = time.time()
        time.sleep(2)
        success = False
        success_selectors = [
            ("xpath", "//*[contains(text(), '成功') or contains(text(), 'Success') or contains(text(), '感谢') or contains(text(), 'Thank')]"),
            ("css_selector", ".success, .order-success, .confirmation"),
            ("css_selector", "[data-testid='order-success']"),
        ]
        for by, selector in success_selectors:
            if manager.safe_find(by, selector, timeout=2):
                success = True
                break

        if success:
            steps.append(TestStepResult(
                step_name="验证提交结果",
                status=TestStatus.PASSED,
                duration_ms=(time.time() - step_start) * 1000,
            ))
            selenium_logs.append("[INFO] 提交成功验证通过")
        else:
            steps.append(TestStepResult(
                step_name="验证提交结果",
                status=TestStatus.FAILED,
                error_message="未检测到提交成功的标志",
            ))
            selenium_logs.append("[WARN] 未检测到提交成功标志")

        all_passed = all(s.status == TestStatus.PASSED for s in steps)
        return _build_order_result(
            steps, start_time, all_passed,
            "" if all_passed else "下单流程存在问题",
            selenium_logs,
        )

    except Exception as e:
        logger.error(f"下单测试异常: {e}", exc_info=True)
        selenium_logs.append(f"[FATAL] {type(e).__name__}: {e}")
        return _build_order_result(steps, start_time, False, str(e), selenium_logs)
    finally:
        manager.quit()


def _build_order_result(
    steps: list[TestStepResult],
    start_time: datetime,
    all_passed: bool,
    error_message: str,
    selenium_logs: list[str],
) -> TestCaseResult:
    """构建下单测试结果"""
    end_time = datetime.now()
    duration_ms = (end_time - start_time).total_seconds() * 1000

    return TestCaseResult(
        scenario="order",
        status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        steps=steps,
        error_message=error_message,
        selenium_logs="\n".join(selenium_logs),
    )
