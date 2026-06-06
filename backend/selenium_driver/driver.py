"""
Selenium WebDriver 管理器模块
提供 Chrome WebDriver 的创建、配置和生命周期管理
"""
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = logging.getLogger("ai_rd_agent")


class WebDriverManager:
    """Chrome WebDriver 管理器

    负责：
    - 创建和配置 Chrome 浏览器实例
    - 提供显式等待封装
    - 管理浏览器生命周期（创建/销毁）

    使用示例：
        manager = WebDriverManager(headless=True)
        driver = manager.create_driver()
        try:
            driver.get("https://example.com")
            manager.wait_for_element("id", "login-btn")
        finally:
            manager.quit()
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_seconds: int = 30,
    ):
        """
        Args:
            headless: 是否使用无头模式（不显示浏览器窗口）
            timeout_seconds: 默认的显式等待超时时间
        """
        self.headless = headless
        self.timeout_seconds = timeout_seconds
        self._driver: webdriver.Chrome | None = None
        self._wait: WebDriverWait | None = None

    def create_driver(self) -> webdriver.Chrome:
        """创建并配置 Chrome WebDriver 实例

        Returns:
            配置好的 Chrome WebDriver
        """
        logger.info(f"创建 Chrome WebDriver (headless={self.headless})")

        options = Options()

        # 无头模式
        if self.headless:
            options.add_argument("--headless=new")  # 新版 headless 模式
            options.add_argument("--disable-gpu")

        # 通用配置
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # 忽略证书错误（测试环境常用）
        options.add_argument("--ignore-certificate-errors")

        try:
            # 使用 Service 对象设置超时
            from selenium.webdriver.chrome.service import Service as ChromeService
            service = ChromeService()
            self._driver = webdriver.Chrome(service=service, options=options)
            self._driver.set_page_load_timeout(self.timeout_seconds)
            self._driver.implicitly_wait(5)  # 隐式等待 5 秒
            self._wait = WebDriverWait(self._driver, self.timeout_seconds)
            logger.info("Chrome WebDriver 创建成功")
            return self._driver
        except Exception as e:
            logger.error(f"WebDriver 创建失败: {e}")
            logger.info("提示：请确保已安装 Chrome 浏览器和 chromedriver")
            logger.info("  pip install webdriver-manager 可以自动管理驱动版本")
            raise

    @property
    def driver(self) -> webdriver.Chrome:
        """获取当前 WebDriver 实例"""
        if self._driver is None:
            raise RuntimeError("WebDriver 尚未创建，请先调用 create_driver()")
        return self._driver

    @property
    def wait(self) -> WebDriverWait:
        """获取显式等待对象"""
        if self._wait is None:
            raise RuntimeError("WebDriver 尚未创建，请先调用 create_driver()")
        return self._wait

    # ==================== 元素等待 ====================

    def wait_for_element(
        self,
        by: str,
        value: str,
        timeout: int | None = None,
    ):
        """等待元素可见并可交互

        Args:
            by: 定位方式（id / name / xpath / css_selector / class_name）
            value: 定位值
            timeout: 超时时间（不传则使用全局配置）

        Returns:
            找到的 WebElement

        Raises:
            TimeoutException: 等待超时
        """
        wait = self.wait if timeout is None else WebDriverWait(self.driver, timeout)
        return wait.until(EC.visibility_of_element_located((by, value)))

    def wait_for_clickable(
        self,
        by: str,
        value: str,
        timeout: int | None = None,
    ):
        """等待元素可被点击

        Args:
            by: 定位方式
            value: 定位值
            timeout: 超时时间

        Returns:
            可点击的 WebElement

        Raises:
            TimeoutException: 等待超时
        """
        wait = self.wait if timeout is None else WebDriverWait(self.driver, timeout)
        return wait.until(EC.element_to_be_clickable((by, value)))

    def safe_find(self, by: str, value: str, timeout: int = 5):
        """安全查找元素 — 找不到返回 None 而非抛异常

        Args:
            by: 定位方式
            value: 定位值
            timeout: 超时秒数

        Returns:
            WebElement 或 None
        """
        try:
            wait = WebDriverWait(self.driver, timeout)
            return wait.until(EC.presence_of_element_located((by, value)))
        except TimeoutException:
            return None

    # ==================== 截图 ====================

    def take_screenshot(self, filename: str) -> str:
        """截取当前页面截图

        Args:
            filename: 截图文件名（不含扩展名）

        Returns:
            截图文件的完整路径
        """
        from pathlib import Path
        from backend.config.settings import get_settings

        settings = get_settings()
        screenshot_dir = Path(settings.PROJECT_ROOT) / "data" / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        filepath = screenshot_dir / f"{filename}.png"
        self.driver.save_screenshot(str(filepath))
        logger.info(f"截图已保存: {filepath}")
        return str(filepath)

    # ==================== 生命周期 ====================

    def quit(self):
        """关闭浏览器并释放资源"""
        if self._driver:
            self._driver.quit()
            self._driver = None
            self._wait = None
            logger.info("WebDriver 已关闭")
