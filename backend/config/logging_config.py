"""
日志配置模块
提供结构化日志配置，支持控制台输出 + 文件轮转，可选 JSON 格式
"""
import logging
import sys
import io
import warnings
from pathlib import Path
from logging.handlers import RotatingFileHandler

from .settings import get_settings
from .json_formatter import JSONFormatter

# 抑制 LangChain 生态中已知的Pending弃用警告（不影响功能）
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except Exception:
    pass


def setup_logging() -> logging.Logger:
    """初始化全局日志配置

    返回根 logger，同时配置控制台和文件两个 handler：
    - 控制台：输出 INFO 及以上级别
    - 文件：输出 DEBUG 及以上级别，自动轮转（单文件最大 10MB，保留 5 个历史文件）

    当 settings.LOG_FORMAT == "json" 时输出 JSON 格式（适用于容器生产和日志聚合系统），
    否则输出人类可读的文本格式。
    """
    settings = get_settings()

    # 确保日志目录存在
    log_dir = Path(settings.get_log_dir())
    log_dir.mkdir(parents=True, exist_ok=True)

    # 获取根 logger
    root_logger = logging.getLogger("ai_rd_agent")
    root_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    # 避免重复添加 handler
    if root_logger.handlers:
        return root_logger

    # ---- 选择格式 ----
    use_json = settings.LOG_FORMAT == "json"

    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # ---- 控制台 Handler（Windows 兼容 UTF-8） ----
    if sys.platform == "win32":
        utf8_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        console_handler = logging.StreamHandler(utf8_stream)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    if use_json:
        # JSON 格式控制台也输出 JSON（便于容器日志收集）
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # ---- 文件轮转 Handler ----
    file_handler = RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # ---- 审计日志 Handler（独立文件，追加写，不轮转） ----
    _setup_audit_logger(log_dir, use_json)

    return root_logger


def _setup_audit_logger(log_dir: Path, use_json: bool):
    """配置审计日志 logger 'audit'，写入独立的 audit.log 文件"""
    audit_logger = logging.getLogger("audit")
    if audit_logger.handlers:
        return

    audit_logger.setLevel(logging.INFO)

    audit_handler = logging.FileHandler(
        filename=log_dir / "audit.log",
        encoding="utf-8",
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(
        JSONFormatter() if use_json else logging.Formatter(
            fmt="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    audit_logger.addHandler(audit_handler)
