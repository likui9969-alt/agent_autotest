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
import time
from dataclasses import dataclass
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, SessionNotCreatedException

logger = logging.getLogger("ai_rd_agent")

# 缓存：Chrome 版本号（进程内缓存，避免每次创建 WebDriver 都调注册表）
_CACHED_CHROME_VERSION: int | None = None
_CACHED_CHROME_FULL_VERSION: str = ""

# chromedriver 缓存条目
_CACHE_TTL_SECONDS = 24 * 3600  # 24 小时


@dataclass
class _CachedDriver:
    """chromedriver 缓存条目 — 包含路径、版本号和缓存时间"""
    path: str
    version: int | None  # chromedriver 主版本号
    cached_at: float     # time.time() 缓存时刻

    def is_valid(self, chrome_version: int | None) -> bool:
        """判断缓存是否仍然有效

        生效条件：
        1. chromedriver 可执行文件仍然存在
        2. 版本与当前 Chrome 主版本匹配（chrome_version 为 None 时跳过）
        3. 缓存未过期（24 小时内）
        """
        if not Path(self.path).exists():
            return False
        if chrome_version is not None and self.version is not None and self.version != chrome_version:
            return False
        if time.time() - self.cached_at > _CACHE_TTL_SECONDS:
            return False
        return True


# 全局缓存：selenium-manager 自动下载的 chromedriver 路径
# 避免每次创建 WebDriver 都触发 selenium-manager 检查（耗时 4-5 分钟）
_CACHED_DRIVER: _CachedDriver | None = None


def _find_cached_chromedriver() -> str:
    """查找 selenium-manager 缓存的 chromedriver 路径

    selenium-manager 下载的 chromedriver 存放在固定缓存目录中。
    如果能找到，直接返回路径，避免重复调用 selenium-manager。
    """
    import os
    import glob
    from pathlib import Path

    # selenium-manager 缓存目录
    if sys.platform == "win32":
        cache_base = Path(os.environ.get("LOCALAPPDATA", "")) / "selenium-manager" / "chromedriver"
    elif sys.platform == "darwin":
        cache_base = Path.home() / "Library" / "Caches" / "selenium-manager" / "chromedriver"
    else:
        cache_base = Path.home() / ".cache" / "selenium-manager" / "chromedriver"

    if not cache_base.exists():
        # 也尝试旧版路径
        if sys.platform == "win32":
            cache_base = Path(os.environ.get("USERPROFILE", "")) / ".cache" / "selenium-manager" / "chromedriver"
        else:
            cache_base = Path.home() / ".cache" / "selenium-manager" / "chromedriver"

    if not cache_base.exists():
        return ""

    # 查找最新的 chromedriver 可执行文件
    patterns = ["**/chromedriver.exe", "**/chromedriver"]
    for pattern in patterns:
        matches = sorted(cache_base.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        if matches:
            return str(matches[0])

    return ""


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


def _get_chrome_major_version(chrome_binary: str = "") -> int | None:
    """获取已安装 Chrome 浏览器的主版本号（结果缓存，仅首次查询注册表）"""
    global _CACHED_CHROME_VERSION
    if _CACHED_CHROME_VERSION is not None:
        return _CACHED_CHROME_VERSION

    import sys

    # Windows：优先注册表（不触发浏览器弹窗）
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["reg", "query", r"HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon", "/v", "version"],
                capture_output=True, text=True, timeout=10,
            )
            match = re.search(r"(\d+)\.", result.stdout)
            if match:
                _CACHED_CHROME_VERSION = int(match.group(1))
                return _CACHED_CHROME_VERSION
        except Exception:
            pass

        # 注册表回退：也查一下 LocalAppData 下的 User Data/Local State
        try:
            lad = os.environ.get("LOCALAPPDATA", "")
            pref_path = os.path.join(lad, "Google", "Chrome", "User Data", "Local State")
            if os.path.exists(pref_path):
                import json
                with open(pref_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                ver = state.get("user_experience_metrics", {}).get("stability", {}).get("browser_version", "")
                if ver:
                    match = re.search(r"(\d+)\.", ver)
                    if match:
                        _CACHED_CHROME_VERSION = int(match.group(1))
                        return _CACHED_CHROME_VERSION
        except Exception:
            pass

    # 非 Windows 或注册表失败：执行 chrome --version（不弹窗的平台）
    binary = chrome_binary or ""
    if not binary and sys.platform == "win32":
        pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        lad = os.environ.get("LOCALAPPDATA", "")
        for candidate in [
            rf"{lad}\Google\Chrome\Application\chrome.exe",
            rf"{pf}\Google\Chrome\Application\chrome.exe",
            rf"{pf86}\Google\Chrome\Application\chrome.exe",
        ]:
            if Path(candidate).exists():
                binary = candidate
                break

    if binary and Path(binary).exists():
        try:
            result = subprocess.run(
                [binary, "--version"],
                capture_output=True, text=True, timeout=15,
            )
            match = re.search(r"(\d+)\.", result.stdout)
            if match:
                _CACHED_CHROME_VERSION = int(match.group(1))
                return _CACHED_CHROME_VERSION
        except Exception:
            pass

    return None


def _get_chrome_full_version(chrome_binary: str = "") -> str:
    """获取 Chrome 完整版本号字符串（结果缓存，仅首次查询注册表）"""
    global _CACHED_CHROME_FULL_VERSION
    if _CACHED_CHROME_FULL_VERSION:
        return _CACHED_CHROME_FULL_VERSION

    import sys

    # Windows：优先注册表（不触发浏览器弹窗）
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["reg", "query", r"HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon", "/v", "version"],
                capture_output=True, text=True, timeout=10,
            )
            match = re.search(r"(\d+\.\d+\.\d+\.\d+)", result.stdout)
            if match:
                _CACHED_CHROME_FULL_VERSION = match.group(1)
                return _CACHED_CHROME_FULL_VERSION
        except Exception:
            pass

    # 非 Windows 或注册表失败：执行 chrome --version
    binary = chrome_binary or ""
    if not binary and sys.platform == "win32":
        lad = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            rf"{lad}\Google\Chrome\Application\chrome.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                binary = c
                break

    if binary and Path(binary).exists():
        try:
            result = subprocess.run(
                [binary, "--version"],
                capture_output=True, text=True, timeout=15,
            )
            match = re.search(r"(\d+\.\d+\.\d+\.\d+)", result.stdout)
            if match:
                _CACHED_CHROME_FULL_VERSION = match.group(1)
                return _CACHED_CHROME_FULL_VERSION
        except Exception:
            pass
    return ""


def _inject_anti_detect(driver: webdriver.Chrome) -> None:
    """通过 CDP 注入脚本隐藏自动化特征"""
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    // 1. 隐藏 webdriver 标记
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

                    // 2. 补全 window.chrome
                    if (!window.chrome) {
                        window.chrome = {};
                    }
                    window.chrome.runtime = window.chrome.runtime || { OnInstalledReason: { CHROME_UPDATE: "chrome_update", EXTENSION_UPDATE: "extension_update", INSTALL: "install" }, OnRestartRequiredReason: { APP_UPDATE: "app_update", OS_UPDATE: "os_update" }, PlatformArch: { ARM: "arm", ARM64: "arm64", MIPS: "mips", MIPS64: "mips64", MIPS64EL: "mips64el", MIPSEL: "mipsel", X86_32: "x86-32", X86_64: "x86-64" }, PlatformNaclArch: { ARM: "arm", MIPS: "mips", MIPS64: "mips64", MIPS64EL: "mips64el", MIPSEL: "mipsel", Mips32: "mips32", Mips64: "mips64", X86_32: "x86-32", X86_64: "x86-64" }, PlatformOs: { ANDROID: "android", CROS: "cros", LINUX: "linux", MAC: "mac", OPENBSD: "openbsd", WIN: "win" }, RequestUpdateCheckStatus: { NO_UPDATE: "no_update", THROTTLED: "throttled", UPDATE_AVAILABLE: "update_available" } };

                    // 3. 模拟插件
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => {
                            return [
                                { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format', version: 'undefined', length: 1, item: function(idx) { return this[idx]; }, namedItem: function(name) { return null; } },
                                { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '', version: 'undefined', length: 1, item: function(idx) { return this[idx]; }, namedItem: function(name) { return null; } },
                                { name: 'Native Client', filename: 'internal-nacl-plugin', description: '', version: 'undefined', length: 2, item: function(idx) { return this[idx]; }, namedItem: function(name) { return null; } }
                            ];
                        }
                    });

                    // 4. 模拟语言
                    Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });

                    // 5. 覆盖 permissions.query，避免被检测无头模式
                    if (navigator.permissions && navigator.permissions.query) {
                        const originalQuery = navigator.permissions.query;
                        navigator.permissions.query = function(parameters) {
                            if (parameters && parameters.name === 'notifications') {
                                return Promise.resolve({ state: Notification.permission });
                            }
                            return originalQuery.call(this, parameters);
                        };
                    }

                    // 6. 覆盖 Notification.permission
                    if (window.Notification) {
                        Object.defineProperty(Notification, 'permission', { get: () => 'default' });
                    }

                    // 7. 修复 iFrame 里的 webdriver 标记
                    try {
                        const iframes = document.getElementsByTagName('iframe');
                        for (var i = 0; i < iframes.length; i++) {
                            var win = iframes[i].contentWindow;
                            if (win && win.navigator) {
                                Object.defineProperty(win.navigator, 'webdriver', { get: () => undefined });
                            }
                        }
                    } catch (e) {}
                """
            }
        )
    except Exception as e:
        logger.debug(f"CDP 反检测脚本注入失败: {e}")


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
        """创建并配置 Chrome WebDriver 实例（幂等：已有实例则直接返回）

        Returns:
            配置好的 Chrome WebDriver
        """
        if self._driver is not None:
            logger.debug("复用已有的 WebDriver 实例")
            return self._driver

        logger.info(f"创建 Chrome WebDriver (headless={self.headless})")

        options = Options()

        # ---- 页面加载策略：eager 表示 DOMContentLoaded 后即返回 ----
        # 避免等待图片/字体等资源加载完成，显著减少 renderer 超时
        options.page_load_strategy = "eager"

        # 无头模式：使用新版 --headless=new（反检测能力更强，行为更接近真实浏览器）
        if self.headless:
            options.add_argument("--headless=new")
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

        # 反检测：使用与 Chrome 版本匹配的 User-Agent
        chrome_full_ver = _get_chrome_full_version(chrome_binary)
        if chrome_full_ver:
            options.add_argument(
                f"--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                f"AppleWebKit/537.36 (KHTML, like Gecko) "
                f"Chrome/{chrome_full_ver} Safari/537.36"
            )

        try:
            # 使用 Service 对象指定 chromedriver 路径
            from selenium.webdriver.chrome.service import Service as ChromeService
            driver_path = settings.get_chromedriver_path()

            # ---- 优先使用全局缓存的 chromedriver 路径 ----
            global _CACHED_DRIVER
            chrome_ver = _get_chrome_major_version(chrome_binary)
            if _CACHED_DRIVER and _CACHED_DRIVER.is_valid(chrome_ver):
                cached_path = _CACHED_DRIVER.path
                logger.info(f"使用缓存的 chromedriver: {cached_path} (v{_CACHED_DRIVER.version})")
                service = ChromeService(executable_path=cached_path)
                try:
                    self._driver = webdriver.Chrome(service=service, options=options)
                except SessionNotCreatedException:
                    # 缓存的驱动可能已失效，清除缓存并继续正常流程
                    logger.warning("缓存的 chromedriver 失效，重新检测")
                    _CACHED_DRIVER = None
                    driver_path = ""
                else:
                    self._driver.set_page_load_timeout(max(self.timeout_seconds, 60))
                    self._driver.implicitly_wait(3)
                    self._wait = WebDriverWait(self._driver, self.timeout_seconds)
                    _inject_anti_detect(self._driver)
                    logger.info("Chrome WebDriver 创建成功（使用缓存驱动）")
                    return self._driver

            # ---- 版本兼容性检查 ----
            # 如果本地 chromedriver 版本与 Chrome 版本不匹配，优先从缓存目录找匹配版本，
            # 否则让 Selenium 4.11+ 内置的 selenium-manager 自动下载匹配版本
            if driver_path:
                drv_ver = _get_chromedriver_major_version(driver_path)
                if drv_ver and chrome_ver and drv_ver != chrome_ver:
                    logger.warning(
                        f"chromedriver 版本不匹配: 本地驱动 v{drv_ver}，Chrome v{chrome_ver}。"
                        f"尝试寻找缓存中的匹配版本或让 Selenium 自动下载。"
                    )
                    # 在缓存目录中查找与 Chrome 主版本匹配的驱动
                    cached = _find_cached_chromedriver()
                    if cached:
                        cached_ver = _get_chromedriver_major_version(cached)
                        if cached_ver == chrome_ver:
                            logger.info(f"找到缓存的匹配驱动 v{cached_ver}: {cached}")
                            _CACHED_DRIVER = _CachedDriver(
                                path=cached, version=cached_ver, cached_at=time.time()
                            )
                            driver_path = cached
                        else:
                            driver_path = ""  # 清空，让 Selenium 自动管理
                    else:
                        driver_path = ""  # 清空，让 Selenium 自动管理
                elif drv_ver and chrome_ver:
                    logger.info(f"chromedriver 版本匹配: 驱动 v{drv_ver}，Chrome v{chrome_ver}")

            if driver_path:
                logger.info(f"使用本地 chromedriver: {driver_path}")
                service = ChromeService(executable_path=driver_path)
            else:
                # 先尝试从 selenium-manager 缓存目录查找已下载的 chromedriver
                cached = _find_cached_chromedriver()
                if cached:
                    cached_ver = _get_chromedriver_major_version(cached)
                    logger.info(f"从缓存目录找到 chromedriver: {cached}")
                    _CACHED_DRIVER = _CachedDriver(
                        path=cached, version=cached_ver, cached_at=time.time()
                    )
                    service = ChromeService(executable_path=cached)
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

            # 缓存成功使用的 chromedriver 路径（如果是自动下载的）
            if not driver_path and not _CACHED_DRIVER:
                # 尝试从 selenium-manager 日志中提取路径，或直接查找缓存目录
                cached = _find_cached_chromedriver()
                if cached:
                    cached_ver = _get_chromedriver_major_version(cached)
                    _CACHED_DRIVER = _CachedDriver(
                        path=cached, version=cached_ver, cached_at=time.time()
                    )
                    logger.info(f"已缓存 chromedriver 路径供后续使用: {cached}")

            self._driver.set_page_load_timeout(max(self.timeout_seconds, 60))
            self._driver.implicitly_wait(3)  # 隐式等待 3 秒（减少以避免与显式等待叠加）
            self._wait = WebDriverWait(self._driver, self.timeout_seconds)
            _inject_anti_detect(self._driver)
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

    # ==================== 页面探索 ====================

    def explore_page(self) -> dict:
        """探索当前页面，提取表单、输入框、按钮等元素信息

        用于 Agent 自适应测试：先打开页面，再根据页面结构决定如何测试。

        Returns:
            {
                "title": 页面标题,
                "url": 当前 URL,
                "page_type": 推测的页面类型 (login/search/unknown),
                "forms": 表单列表,
                "inputs": 输入框列表,
                "buttons": 按钮列表,
                "links": 链接列表,
            }
        """
        from selenium.webdriver.common.by import By

        result = {
            "title": self.driver.title or "",
            "url": self.driver.current_url or "",
            "page_type": "unknown",
            "forms": [],
            "inputs": [],
            "buttons": [],
            "links": [],
        }

        # 提取输入框
        try:
            inputs = self.driver.find_elements(By.TAG_NAME, "input")
            for inp in inputs:
                inp_type = inp.get_attribute("type") or "text"
                result["inputs"].append({
                    "type": inp_type,
                    "name": inp.get_attribute("name") or "",
                    "id": inp.get_attribute("id") or "",
                    "placeholder": inp.get_attribute("placeholder") or "",
                    "by": "id" if inp.get_attribute("id") else ("name" if inp.get_attribute("name") else "css_selector"),
                    "value": inp.get_attribute("id") or inp.get_attribute("name") or f"input[type='{inp_type}']",
                })
        except Exception:
            pass

        # 提取按钮
        try:
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons:
                result["buttons"].append({
                    "text": (btn.text or "").strip()[:50],
                    "type": btn.get_attribute("type") or "button",
                    "id": btn.get_attribute("id") or "",
                    "name": btn.get_attribute("name") or "",
                    "by": "id" if btn.get_attribute("id") else "xpath",
                    "value": btn.get_attribute("id") or f"//button[contains(text(),'{(btn.text or '').strip()[:20]}')]",
                })
        except Exception:
            pass

        # 提取链接（最多 20 个）
        try:
            links = self.driver.find_elements(By.TAG_NAME, "a")
            for link in links[:20]:
                result["links"].append({
                    "text": (link.text or "").strip()[:50],
                    "href": link.get_attribute("href") or "",
                })
        except Exception:
            pass

        # 推测页面类型
        title_lower = result["title"].lower()
        url_lower = result["url"].lower()
        input_types = [i["type"] for i in result["inputs"]]
        button_texts = [b["text"].lower() for b in result["buttons"]]

        has_password = "password" in input_types
        has_login_keyword = any(
            kw in title_lower or kw in url_lower or any(kw in bt for bt in button_texts)
            for kw in ["登录", "login", "sign in", "log in"]
        )
        has_search_keyword = any(
            kw in title_lower or kw in url_lower
            for kw in ["搜索", "search", "百度", "google"]
        )

        if has_password and has_login_keyword:
            result["page_type"] = "login"
        elif has_search_keyword or (len(result["inputs"]) == 1 and result["inputs"][0]["type"] == "text"):
            result["page_type"] = "search"
        else:
            result["page_type"] = "unknown"

        return result

    # ==================== 生命周期 ====================

    def quit(self):
        """关闭浏览器并释放资源"""
        if self._driver:
            self._driver.quit()
            self._driver = None
            self._wait = None
            logger.info("WebDriver 已关闭")
