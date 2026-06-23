"""
Selenium WebDriver 管理器模块
提供 Chrome WebDriver 的创建、配置和生命周期管理
"""
import logging
import re
import os
import sys
import shutil
import subprocess
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, SessionNotCreatedException

logger = logging.getLogger("ai_rd_agent")


def detect_chrome() -> tuple[str, str]:
    """检测 Chrome 浏览器和 chromedriver 路径（轻量检测，不启动浏览器）

    Returns:
        (chrome_binary_path, chromedriver_path) — 未找到则为空字符串
    """
    from backend.config.settings import get_settings

    settings = get_settings()

    # ---- 1. 检测 chromedriver ----
    driver_path = settings.get_chromedriver_path()
    if not driver_path:
        driver_path = shutil.which("chromedriver") or shutil.which("chromedriver.exe") or ""

    # ---- 2. 检测 Chrome 浏览器 ----
    chrome_binary = settings.CHROME_BINARY_PATH
    if chrome_binary:
        if not Path(chrome_binary).exists():
            chrome_binary = ""
    else:
        if sys.platform == "win32":
            pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
            pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
            lad = os.environ.get("LOCALAPPDATA", "")
            candidates = [
                rf"{pf}\Google\Chrome\Application\chrome.exe",
                rf"{pf86}\Google\Chrome\Application\chrome.exe",
                rf"{lad}\Google\Chrome\Application\chrome.exe",
            ]
        else:
            candidates = [
                "/usr/bin/google-chrome",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium",
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ]
        chrome_binary = next((p for p in candidates if Path(p).exists()), "")

    return chrome_binary, driver_path

# 用户友好字符串 → Selenium By 常量的映射
# 解决 "css_selector"（下划线）vs Selenium 要求 "css selector"（空格）的兼容问题
_BY_MAP = {
    "id": By.ID,
    "name": By.NAME,
    "xpath": By.XPATH,
    "css_selector": By.CSS_SELECTOR,
    "css selector": By.CSS_SELECTOR,
    "css": By.CSS_SELECTOR,
    "class_name": By.CLASS_NAME,
    "class name": By.CLASS_NAME,
    "class": By.CLASS_NAME,
    "tag_name": By.TAG_NAME,
    "tag name": By.TAG_NAME,
    "tag": By.TAG_NAME,
    "link_text": By.LINK_TEXT,
    "link text": By.LINK_TEXT,
    "partial_link_text": By.PARTIAL_LINK_TEXT,
    "partial link text": By.PARTIAL_LINK_TEXT,
}


def _resolve_by(by: str) -> str:
    """将用户友好的定位方式字符串解析为 Selenium 认可的 By 常量字符串

    兼容以下写法（不区分大小写）：
        "css_selector" / "css selector" / "css"  → By.CSS_SELECTOR
        "class_name"  / "class name"  / "class"  → By.CLASS_NAME
        "tag_name"    / "tag name"    / "tag"    → By.TAG_NAME
        "id" / "name" / "xpath" / "link_text" / "partial_link_text"

    Args:
        by: 定位方式字符串

    Returns:
        Selenium By 常量字符串

    Raises:
        ValueError: 不支持的定位方式
    """
    key = by.strip().lower()
    if key not in _BY_MAP:
        raise ValueError(
            f"不支持的定位方式: '{by}'。"
            f"支持: id / name / xpath / css_selector / class_name / tag_name / link_text / partial_link_text"
        )
    return _BY_MAP[key]


def _get_chromedriver_major_version(driver_path: str) -> int | None:
    """获取 chromedriver 的主版本号

    Args:
        driver_path: chromedriver 可执行文件路径

    Returns:
        主版本号（如 146），获取失败返回 None
    """
    try:
        result = subprocess.run(
            [driver_path, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        # 输出格式: ChromeDriver 146.0.7680.2 (...)
        match = re.search(r"(\d+)\.", result.stdout)
        return int(match.group(1)) if match else None
    except Exception:
        return None


def _get_chrome_major_version() -> int | None:
    """获取已安装 Chrome 浏览器的主版本号（Windows 注册表）

    Returns:
        主版本号（如 148），获取失败返回 None
    """
    import sys
    if sys.platform != "win32":
        return None
    try:
        result = subprocess.run(
            ["reg", "query", r"HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon", "/v", "version"],
            capture_output=True, text=True, timeout=10,
        )
        # 输出含: version    REG_SZ    148.0.7778.97
        match = re.search(r"(\d+)\.", result.stdout)
        return int(match.group(1)) if match else None
    except Exception:
        return None


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

        # ---- 页面加载策略：eager 表示 DOMContentLoaded 后即返回 ----
        # 避免等待图片/字体等资源加载完成，显著减少 renderer 超时
        options.page_load_strategy = "eager"

        # 无头模式：使用旧版 --headless（比 --headless=new 更稳定，renderer 超时更少）
        if self.headless:
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")

        # ---- 稳定性优化参数（解决 Windows 下 renderer 超时） ----
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-sync")
        options.add_argument("--no-first-run")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-features=TranslateUI,VizDisplayCompositor,IsolateOrigins,site-per-process")
        options.add_argument("--remote-debugging-port=0")
        options.add_argument("--password-store=basic")
        options.add_argument("--use-mock-keychain")
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option("useAutomationExtension", False)

        # 忽略证书错误（测试环境常用）
        options.add_argument("--ignore-certificate-errors")

        # 指定 Chrome 浏览器可执行文件路径（如配置）
        from backend.config.settings import get_settings
        settings = get_settings()
        chrome_binary = settings.CHROME_BINARY_PATH
        if chrome_binary:
            options.binary_location = chrome_binary

        try:
            # 使用 Service 对象指定 chromedriver 路径
            from selenium.webdriver.chrome.service import Service as ChromeService
            driver_path = settings.get_chromedriver_path()

            # ---- 版本兼容性检查 ----
            # 如果本地 chromedriver 版本与 Chrome 版本不匹配，跳过本地驱动，
            # 让 Selenium 4.11+ 内置的 selenium-manager 自动下载匹配版本
            if driver_path:
                drv_ver = _get_chromedriver_major_version(driver_path)
                chrome_ver = _get_chrome_major_version()
                if drv_ver and chrome_ver and drv_ver != chrome_ver:
                    logger.warning(
                        f"chromedriver 版本不匹配: 本地驱动 v{drv_ver}，Chrome v{chrome_ver}。"
                        f"跳过本地驱动，使用 Selenium 自动下载匹配版本。"
                    )
                    driver_path = ""  # 清空，让 Selenium 自动管理
                elif drv_ver and chrome_ver:
                    logger.info(f"chromedriver 版本匹配: 驱动 v{drv_ver}，Chrome v{chrome_ver}")

            if driver_path:
                logger.info(f"使用本地 chromedriver: {driver_path}")
                service = ChromeService(executable_path=driver_path)
            else:
                logger.info("使用 Selenium 内置驱动管理器自动下载 chromedriver")
                service = ChromeService()

            try:
                self._driver = webdriver.Chrome(service=service, options=options)
            except SessionNotCreatedException as e:
                # 版本不匹配导致 session 创建失败，回退到自动下载
                if driver_path and "only supports" in str(e):
                    logger.warning(
                        f"本地 chromedriver 版本不兼容，回退到 Selenium 自动下载: {e}"
                    )
                    service = ChromeService()  # 不指定路径，触发自动下载
                    self._driver = webdriver.Chrome(service=service, options=options)
                else:
                    raise

            self._driver.set_page_load_timeout(max(self.timeout_seconds, 60))
            self._driver.implicitly_wait(3)  # 隐式等待 3 秒（减少以避免与显式等待叠加）
            self._wait = WebDriverWait(self._driver, self.timeout_seconds)
            logger.info("Chrome WebDriver 创建成功")
            return self._driver
        except Exception as e:
            logger.error(f"WebDriver 创建失败: {e}")
            logger.info("提示：请确保已安装 Chrome 浏览器；Selenium 4.11+ 可自动下载匹配的 chromedriver")
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

    # ==================== 页面加载 ====================

    def safe_get(self, url: str, retries: int = 2) -> None:
        """安全页面加载 — 处理 Chrome renderer 超时

        Chrome headless 在 Windows 下经常报 "Timed out receiving message from renderer"，
        但实际上页面已经加载完成。此方法捕获该异常并验证页面是否真的可用。

        Args:
            url: 要访问的 URL
            retries: 重试次数

        Raises:
            TimeoutException: 重试后仍无法加载页面
        """
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                self.driver.get(url)
                return
            except TimeoutException as e:
                last_exc = e
                msg = str(e)
                # renderer 超时通常页面已加载，验证 document.readyState
                if "renderer" in msg or "Timed out receiving message" in msg:
                    try:
                        ready = self.driver.execute_script("return document.readyState")
                        cur_url = self.driver.current_url
                        if ready in ("interactive", "complete") and cur_url:
                            logger.warning(
                                f"页面加载报 renderer 超时，但实际已加载 "
                                f"(readyState={ready}, url={cur_url})，继续执行"
                            )
                            return
                    except Exception as e:
                        logger.debug(f"检查 document.readyState 失败: {e}")
                    logger.warning(
                        f"页面加载 renderer 超时 (尝试 {attempt}/{retries}): {url}"
                    )
                    if attempt < retries:
                        import time as _time
                        _time.sleep(1)
                        continue
                # 非 renderer 超时，直接抛出
                raise
        if last_exc:
            raise last_exc

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
        return wait.until(EC.visibility_of_element_located((_resolve_by(by), value)))

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
        return wait.until(EC.element_to_be_clickable((_resolve_by(by), value)))

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
            return wait.until(EC.presence_of_element_located((_resolve_by(by), value)))
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
        from backend.config.settings import PROJECT_ROOT

        screenshot_dir = PROJECT_ROOT / "data" / "screenshots"
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
