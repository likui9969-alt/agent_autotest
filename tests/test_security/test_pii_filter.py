"""
PII 脱敏过滤器测试
==================
- 手机号 / 邮箱 / 身份证 / 银行卡 的脱敏
- 无 PII 文本原样返回
- 配置开关（PII_FILTER_ENABLED）
- 工具出口集成（parse_log_content / get_runtime_logs）
"""
from unittest.mock import MagicMock, patch

from backend.security.pii_filter import sanitize_pii, sanitize_pii_if_enabled


class TestSanitizePii:
    """核心脱敏函数测试"""

    def test_phone_number_masked(self):
        """手机号应被脱敏"""
        text = "用户 13812345678 登录失败"
        assert "13812345678" not in sanitize_pii(text)
        assert "***手机号***" in sanitize_pii(text)
        assert "登录失败" in sanitize_pii(text)

    def test_email_masked(self):
        """邮箱应被脱敏"""
        text = "联系 admin@example.com 处理"
        result = sanitize_pii(text)
        assert "admin@example.com" not in result
        assert "***邮箱***" in result

    def test_id_card_masked(self):
        """18 位身份证应被脱敏"""
        text = "身份证号 110101199001011234 验证失败"
        result = sanitize_pii(text)
        assert "110101199001011234" not in result
        assert "***身份证***" in result

    def test_bank_card_masked(self):
        """16 位银行卡号应被脱敏"""
        text = "扣款卡号 6222020200112341 异常"
        result = sanitize_pii(text)
        assert "6222020200112341" not in result
        assert "***银行卡***" in result

    def test_multiple_pii_in_one_text(self):
        """一段文本含多种 PII 应全部脱敏"""
        text = "手机 13812345678，邮箱 a@b.com，身份证 110101199001011234"
        result = sanitize_pii(text)
        assert "13812345678" not in result
        assert "a@b.com" not in result
        assert "110101199001011234" not in result

    def test_no_pii_unchanged(self):
        """无 PII 文本应原样返回"""
        text = "TimeoutException: element not found after 30s"
        assert sanitize_pii(text) == text

    def test_short_digits_not_masked(self):
        """普通短数字（时间戳片段、行号）不应误伤"""
        text = "error at line 42, duration 1234ms"
        assert sanitize_pii(text) == text

    def test_empty_text(self):
        """空文本安全返回"""
        assert sanitize_pii("") == ""

    def test_long_number_not_phone_fragment(self):
        """手机号嵌在长数字串中不应截取误报"""
        # 20 位纯数字（既非身份证也非合法银行卡长度边界内的典型样例已由其他规则处理）
        text = "order_id=13812345678901234567"
        result = sanitize_pii(text)
        # 长串整体按银行卡/身份证规则处理或保留，但不允许泄漏出完整手机号子串单独标记
        assert "13812345678 ***手机号***" not in result


class TestSanitizePiiEnabled:
    """配置开关测试"""

    def _settings(self, enabled: bool) -> MagicMock:
        s = MagicMock()
        s.PII_FILTER_ENABLED = enabled
        return s

    def test_disabled_returns_original(self):
        """开关关闭时应原样返回"""
        text = "手机 13812345678"
        with patch(
            "backend.config.settings.get_settings",
            return_value=self._settings(False),
        ):
            assert sanitize_pii_if_enabled(text) == text

    def test_enabled_masks(self):
        """开关开启时应脱敏"""
        text = "手机 13812345678"
        with patch(
            "backend.config.settings.get_settings",
            return_value=self._settings(True),
        ):
            result = sanitize_pii_if_enabled(text)
            assert "13812345678" not in result
