"""
JSON 结构化日志格式化器
输出 logstash 兼容的 JSON 格式，便于日志聚合系统（ELK、Loki 等）解析。
支持通过 contextvars 携带 request_id / trace_id 等上下文。
"""
import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

# 请求 ID 上下文变量，在中间件中设置
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JSONFormatter(logging.Formatter):
    """JSON 结构化日志格式化器

    将日志记录格式化为一行 JSON，包含：
    - @timestamp: ISO8601 时间戳
    - level: 日志级别
    - logger: logger 名称
    - module: 模块名:行号
    - message: 日志消息
    - request_id: 请求 ID（如有）
    - exception: 异常堆栈（如有）
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": f"{record.module}:{record.lineno}",
            "message": record.getMessage(),
        }

        # 附加 contextvars 中的 request_id
        rid = request_id_var.get()
        if rid:
            log_entry["request_id"] = rid

        # 异常信息
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)
