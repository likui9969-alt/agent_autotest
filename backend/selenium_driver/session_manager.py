"""
Selenium Session 管理器

同一批次测试共享单个 WebDriver 实例，每条用例执行前隔离浏览器状态：
- delete_all_cookies()
- localStorage.clear()
- sessionStorage.clear()
- 导航到 base_url

批次结束后统一调用 end_session() 释放资源。
"""
import logging
from typing import Any

from backend.selenium_driver.driver import WebDriverManager

logger = logging.getLogger("ai_rd_agent")


class SessionManager:
    """管理同批次测试共享的 WebDriver 会话

    用法：
        manager = SessionManager(headless=True, timeout_seconds=30)
        manager.create_driver()
        manager.isolate("https://example.com")
        # ... 执行用例 1
        manager.isolate("https://example.com")
        # ... 执行用例 2
        manager.end_session()
    """

    def __init__(self, headless: bool = True, timeout_seconds: int = 30, **kwargs):
        self._headless = headless
        self._timeout_seconds = timeout_seconds
        self._kwargs = kwargs
        self._manager = WebDriverManager(
            headless=headless,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        self._driver = None

    def create_driver(self):
        """创建或复用 WebDriver 实例"""
        if self._driver is None:
            self._driver = self._manager.create_driver()
        return self._driver

    def get_driver(self):
        """获取当前 WebDriver 实例，不存在则创建"""
        if self._driver is None:
            self.create_driver()
        return self._driver

    def isolate(self, base_url: str) -> None:
        """用例间状态隔离

        清除 cookies、localStorage、sessionStorage，并导航到首页，
        避免上一用例的登录态/表单数据影响下一用例。
        """
        driver = self.get_driver()
        try:
            driver.delete_all_cookies()
        except Exception as e:
            logger.warning(f"清除 cookies 失败: {e}")

        try:
            driver.execute_script("window.localStorage.clear();")
            driver.execute_script("window.sessionStorage.clear();")
        except Exception as e:
            logger.warning(f"清除 storage 失败: {e}")

        try:
            self._manager.safe_get(base_url.rstrip("/") + "/")
        except Exception as e:
            logger.warning(f"隔离导航失败: {e}")

    def end_session(self) -> None:
        """结束整个批次会话，释放 WebDriver"""
        if self._manager is not None:
            try:
                self._manager.quit()
            except Exception as e:
                logger.warning(f"关闭 WebDriver 失败: {e}")
        self._driver = None

    # 透传 WebDriverManager 的常用方法，保持对现有 scenario 的兼容
    def safe_get(self, url: str, retries: int = 2) -> None:
        return self._manager.safe_get(url, retries=retries)

    def wait_for_element(self, by: str, value: str, timeout: int = 10):
        return self._manager.wait_for_element(by, value, timeout=timeout)

    def safe_find(self, by: str, value: str, timeout: int = 5):
        return self._manager.safe_find(by, value, timeout=timeout)

    def __getattr__(self, name: str) -> Any:
        """其他未显式实现的方法全部透传给内部 WebDriverManager"""
        return getattr(self._manager, name)
