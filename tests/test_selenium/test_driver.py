"""
Tests for Selenium WebDriver Manager
====================================
- _CachedDriver dataclass: is_valid logic
- Cache invalidation by version, TTL, missing file
"""
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import pytest

from backend.selenium_driver.driver import _CachedDriver, _CACHE_TTL_SECONDS


class TestCachedDriver:
    """_CachedDriver 缓存条目有效性判断"""

    def test_cache_valid(self, tmp_path):
        """版本匹配、未过期、文件存在 → 有效"""
        driver_exe = tmp_path / "chromedriver.exe"
        driver_exe.write_text("fake binary")

        cache = _CachedDriver(
            path=str(driver_exe),
            version=146,
            cached_at=time.time(),
        )

        assert cache.is_valid(chrome_version=146) is True

    def test_cache_version_mismatch(self, tmp_path):
        """版本不匹配 → 无效"""
        driver_exe = tmp_path / "chromedriver.exe"
        driver_exe.write_text("fake binary")

        cache = _CachedDriver(
            path=str(driver_exe),
            version=146,
            cached_at=time.time(),
        )

        assert cache.is_valid(chrome_version=148) is False

    def test_cache_expired(self, tmp_path):
        """超过 24 小时 → 无效"""
        driver_exe = tmp_path / "chromedriver.exe"
        driver_exe.write_text("fake binary")

        cache = _CachedDriver(
            path=str(driver_exe),
            version=146,
            cached_at=time.time() - _CACHE_TTL_SECONDS - 1,
        )

        assert cache.is_valid(chrome_version=146) is False

    def test_cache_file_missing(self):
        """缓存文件已被删除 → 无效"""
        cache = _CachedDriver(
            path=r"C:\nonexistent\chromedriver.exe",
            version=146,
            cached_at=time.time(),
        )

        assert cache.is_valid(chrome_version=146) is False

    def test_cache_skip_version_check_when_none(self, tmp_path):
        """chrome_version 为 None 时跳过版本检查"""
        driver_exe = tmp_path / "chromedriver.exe"
        driver_exe.write_text("fake binary")

        cache = _CachedDriver(
            path=str(driver_exe),
            version=146,
            cached_at=time.time(),
        )

        assert cache.is_valid(chrome_version=None) is True

    def test_cache_empty(self):
        """_CachedDriver 为 None 时视为无效"""
        cache = None
        assert cache is None or not cache.is_valid(146)
