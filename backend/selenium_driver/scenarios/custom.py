"""
自定义测试场景执行器
根据用户定义的步骤列表驱动 Selenium 执行任意网站测试
"""
import time
import random
import logging
from datetime import datetime

from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.common.action_chains import ActionChains

from backend.models.testing import (
    TestRunRequest,
    TestCaseResult,
    TestStepResult,
    TestStatus,
    CustomScenario,
    CustomStepAction,
)
from backend.selenium_driver.driver import WebDriverManager

logger = logging.getLogger("ai_rd_agent")


def run_custom_scenario(
    request: TestRunRequest,
    scenario: CustomScenario,
    shared_manager: WebDriverManager | None = None,
) -> TestCaseResult:
    """执行自定义测试场景

    Args:
        request: 测试请求（含 base_url、headless、timeout 等配置）
        scenario: 自定义场景定义（含步骤列表）
        shared_manager: 可选的共享浏览器实例

    Returns:
        测试用例执行结果
    """
    start_time = datetime.now()
    steps = []
    selenium_logs = []
    own_manager = False
    if shared_manager is None:
        manager = WebDriverManager(
            headless=request.headless,
            timeout_seconds=request.timeout_seconds,
        )
        own_manager = True
    else:
        manager = shared_manager

    try:
        driver = manager.create_driver()
        base_url = request.base_url.rstrip("/")
        selenium_logs.append(f"[INFO] 开始自定义场景: {scenario.name}")

        for i, step_def in enumerate(scenario.steps):
            step_name = step_def.description or f"步骤{i+1}: {step_def.action.value} {step_def.value[:50]}"
            step_start = time.time()

            try:
                if step_def.action == CustomStepAction.NAVIGATE:
                    # 打开 URL（支持相对路径和绝对路径）
                    url = step_def.value
                    if not url.startswith("http"):
                        url = f"{base_url}{url}"
                    manager.safe_get(url)
                    selenium_logs.append(f"[INFO] 导航到 {url}")

                elif step_def.action == CustomStepAction.INPUT:
                    # 在指定元素中输入文本（支持 JS 回退，处理隐藏或被反爬遮蔽的元素）
                    locator, input_text = _split_input_value(step_def.value)
                    el = _safe_find_element(manager, step_def.by, locator, timeout=10)
                    if el is None:
                        raise NoSuchElementException(f"未找到输入元素: {step_def.by}={locator}")
                    _safe_input(manager, el, input_text)
                    _human_delay(200, 600)
                    selenium_logs.append(f"[INFO] 输入文本到 {step_def.by}={locator}")

                elif step_def.action == CustomStepAction.CLICK:
                    # 点击指定元素（支持 JS 点击回退）
                    el = _safe_find_element(manager, step_def.by, step_def.value, timeout=10)
                    if el is None:
                        raise NoSuchElementException(f"未找到点击元素: {step_def.by}={step_def.value}")
                    _safe_click(manager, el)
                    _human_delay(300, 800)
                    selenium_logs.append(f"[INFO] 点击 {step_def.by}={step_def.value[:50]}")

                elif step_def.action == CustomStepAction.VERIFY:
                    # 验证元素存在（presence 即可，不要求可见）
                    el = _safe_find_element(manager, step_def.by, step_def.value, timeout=10)
                    if el is None:
                        raise NoSuchElementException(f"未找到元素: {step_def.by}={step_def.value}")
                    selenium_logs.append(f"[INFO] 验证元素存在: {step_def.by}={step_def.value[:50]}")
                    if not el.is_displayed():
                        selenium_logs.append(f"[WARN] 验证元素存在但不可见: {step_def.by}={step_def.value[:50]}")

                elif step_def.action == CustomStepAction.WAIT:
                    # 等待指定秒数
                    wait_sec = float(step_def.value) if step_def.value else 1.0
                    time.sleep(wait_sec)
                    selenium_logs.append(f"[INFO] 等待 {wait_sec}s")

                # 检查是否命中反爬/验证码页面
                _check_captcha(manager, selenium_logs)

                steps.append(TestStepResult(
                    step_name=step_name,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - step_start) * 1000,
                ))

            except (TimeoutException, NoSuchElementException) as e:
                steps.append(TestStepResult(
                    step_name=step_name,
                    status=TestStatus.FAILED,
                    duration_ms=(time.time() - step_start) * 1000,
                    error_message=str(e)[:300],
                ))
                selenium_logs.append(f"[ERROR] 步骤失败: {step_name} — {e}")
                # 失败后截图
                try:
                    screenshot_path = manager.take_screenshot(f"custom_{scenario.name}_{i+1}_fail")
                    steps[-1].screenshot_path = screenshot_path
                except Exception:
                    pass
                # 继续执行下一步（不中断）
                continue

            except Exception as e:
                steps.append(TestStepResult(
                    step_name=step_name,
                    status=TestStatus.ERROR,
                    duration_ms=(time.time() - step_start) * 1000,
                    error_message=str(e)[:300],
                ))
                selenium_logs.append(f"[ERROR] 步骤异常: {step_name} — {e}")
                continue

        all_passed = all(s.status == TestStatus.PASSED for s in steps) if steps else False
        return _build_result(
            scenario.name, steps, start_time, all_passed,
            "" if all_passed else "部分步骤执行失败",
            selenium_logs,
        )

    except Exception as e:
        logger.error(f"自定义场景执行异常: {e}", exc_info=True)
        selenium_logs.append(f"[FATAL] {type(e).__name__}: {e}")
        steps.append(TestStepResult(
            step_name="测试执行",
            status=TestStatus.ERROR,
            error_message=str(e),
        ))
        return _build_result(scenario.name, steps, start_time, False, str(e), selenium_logs)
    finally:
        if own_manager:
            manager.quit()


def _split_input_value(value: str) -> tuple[str, str]:
    """解析 input 步骤的值，格式：定位值::输入文本"""
    if "::" in value:
        locator, text = value.split("::", 1)
        return locator.strip(), text
    return value.strip(), ""


def _safe_find_element(manager: WebDriverManager, by: str, value: str, timeout: int = 10):
    """安全查找元素：使用 presence，找不到返回 None"""
    return manager.safe_find(by, value, timeout=timeout)


def _human_delay(min_ms: float = 200, max_ms: float = 800) -> None:
    """模拟人类操作间隔的随机延迟"""
    time.sleep(random.uniform(min_ms, max_ms) / 1000.0)


def _is_element_usable(el) -> bool:
    """判断元素是否可见且可交互"""
    try:
        return el.is_displayed() and el.size.get("height", 0) > 0 and el.size.get("width", 0) > 0
    except Exception:
        return False


def _safe_input(manager: WebDriverManager, el, text: str) -> None:
    """安全输入：优先模拟真实用户输入，失败再回退 JS 设置 value"""
    driver = manager.driver
    last_err = None

    # 1) 原生 clear + send_keys
    try:
        if _is_element_usable(el):
            el.clear()
            el.send_keys(text)
            return
    except ElementNotInteractableException as e:
        last_err = e

    # 2) ActionChains 模拟鼠标聚焦后输入（对 Baidu 等反爬页面更友好）
    try:
        ActionChains(driver).move_to_element(el).click(el).send_keys(text).perform()
        return
    except Exception as e:
        last_err = e

    # 3) 最终回退：JS 设置 value 并触发 input/change
    # 使用 KeyboardEvent/InputEvent 更贴近真实输入，且兼容不同 Chrome 版本
    driver.execute_script(
        """
        var el = arguments[0];
        var text = arguments[1];
        el.focus();
        el.value = text;
        el.setAttribute('value', text);
        ['focus', 'input', 'change', 'keyup'].forEach(function(evt) {
            var event;
            try {
                if (evt === 'input') {
                    event = new InputEvent(evt, { bubbles: true, data: text, inputType: 'insertText' });
                } else if (evt === 'keyup') {
                    event = new KeyboardEvent(evt, { bubbles: true, key: text.slice(-1) || '' });
                } else {
                    event = new Event(evt, { bubbles: true });
                }
            } catch (e) {
                event = document.createEvent('Event');
                event.initEvent(evt, true, true);
            }
            el.dispatchEvent(event);
        });
        el.blur();
        """,
        el,
        text,
    )


def _safe_click(manager: WebDriverManager, el) -> None:
    """安全点击：优先原生点击，其次 ActionChains，最后 JS 点击"""
    driver = manager.driver
    try:
        if _is_element_usable(el):
            el.click()
            return
    except ElementNotInteractableException:
        pass

    try:
        ActionChains(driver).move_to_element(el).click(el).perform()
        return
    except Exception:
        pass

    driver.execute_script("arguments[0].click();", el)


def _check_captcha(manager: WebDriverManager, selenium_logs: list[str]) -> None:
    """检测是否进入验证码/反爬拦截页面，给出明确错误"""
    try:
        url = manager.driver.current_url.lower()
        title = manager.driver.title.lower()
    except Exception:
        return
    captcha_signs = ["captcha", "wappass", "verify", "challenge", "cloudflare", "security check"]
    if any(sign in url or sign in title for sign in captcha_signs):
        msg = f"页面触发反爬/验证码拦截（url={manager.driver.current_url}），建议切换为可见模式或更换目标网站"
        selenium_logs.append(f"[WARN] {msg}")
        raise RuntimeError(msg)


def _build_result(
    scenario_name: str,
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
        scenario=scenario_name,
        status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        steps=steps,
        error_message=error_message,
        selenium_logs="\n".join(selenium_logs),
    )
