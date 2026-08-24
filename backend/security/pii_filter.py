"""
PII 脱敏过滤器
==============

对流向 LLM / Agent 上下文的日志与文本做个人敏感信息（PII）脱敏，
避免手机号、邮箱、身份证号、银行卡号进入 Prompt 或持久化报告。

应用点（工具出口，见 agent/tools/__init__.py）：
- parse_log_content / get_runtime_logs 的返回文本

替换策略：保留类型标记便于人工排查，如
- 13812345678   → ***手机号***
- a@b.com       → ***邮箱***
- 110101199001011234 → ***身份证***
- 6222020200112341234 → ***银行卡***

开关：settings.PII_FILTER_ENABLED（默认开启）。
"""
import logging
import re

logger = logging.getLogger("ai_rd_agent")

# 手机号：1[3-9] 开头 11 位，前后不能紧跟数字（避免从长数字串中截取）
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# 邮箱
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# 身份证：18 位（末位可为 X），前后不能紧跟数字/字母X
_ID_CARD_RE = re.compile(r"(?<![\dXx])\d{17}[\dXx](?![\dXx])")

# 银行卡：16~19 位连续数字，前后不能紧跟数字
# 4 位一组（可选空格分隔）的形式也覆盖：\d{4}(?: ?\d{4}){3}(?: ?\d{1,3})?
_BANK_CARD_RE = re.compile(r"(?<!\d)(?:\d{4}(?: ?\d{4}){3}(?: ?\d{1,3})?|\d{16,19})(?!\d)")


def sanitize_pii(text: str) -> str:
    """脱敏文本中的 PII（手机号/邮箱/身份证/银行卡）

    无 PII 时原样返回（不拷贝语义，直接返回入参引用）。
    """
    if not text:
        return text

    result = _ID_CARD_RE.sub("***身份证***", text)
    result = _BANK_CARD_RE.sub("***银行卡***", result)
    result = _PHONE_RE.sub("***手机号***", result)
    result = _EMAIL_RE.sub("***邮箱***", result)

    if result != text:
        logger.info("PII 过滤器已对输出脱敏")
    return result


def sanitize_pii_if_enabled(text: str) -> str:
    """按配置开关脱敏（工具出口统一走此入口）"""
    from backend.config import settings as _settings_module

    if not _settings_module.get_settings().PII_FILTER_ENABLED:
        return text
    return sanitize_pii(text)
