"""
全局异常处理模块
统一捕获和格式化 API 异常响应
"""
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("ai_rd_agent")


class AppException(Exception):
    """应用自定义异常基类"""

    def __init__(self, message: str, status_code: int = 500, detail: dict | None = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail or {}
        super().__init__(message)


class NotFoundException(AppException):
    """资源未找到异常"""

    def __init__(self, message: str = "请求的资源不存在"):
        super().__init__(message, status_code=404)


class ValidationException(AppException):
    """数据校验异常"""

    def __init__(self, message: str = "数据校验失败"):
        super().__init__(message, status_code=422)


class LLMException(AppException):
    """LLM 调用异常"""

    def __init__(self, message: str = "大模型调用失败"):
        super().__init__(message, status_code=502)


class RAGException(AppException):
    """RAG 管线异常"""

    def __init__(self, message: str = "知识库操作失败"):
        super().__init__(message, status_code=500)


def register_exception_handlers(app: FastAPI):
    """注册全局异常处理器到 FastAPI 应用"""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        """处理所有自定义应用异常"""
        logger.warning(f"应用异常: {exc.message} (status={exc.status_code})")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """兜底异常处理器 — 捕获所有未被处理的异常"""
        logger.error(f"未处理异常: {type(exc).__name__}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "服务器内部错误",
                "detail": {"type": type(exc).__name__},
            },
        )
