"""
依赖注入模块
提供 FastAPI 路由中使用的共享依赖（LLM 客户端、RAG 管线等）
各依赖采用懒加载 + 单例模式，避免重复初始化
"""
from functools import lru_cache


@lru_cache()
def get_settings():
    """获取应用配置（延迟导入避免循环依赖）"""
    from backend.config.settings import get_settings as _get_settings
    return _get_settings()


def get_llm_client():
    """获取 LLM 客户端单例"""
    from backend.llm.client import LLMClient
    return LLMClient()


def get_rag_pipeline():
    """获取 RAG 管线单例"""
    from backend.rag.pipeline import RAGPipeline
    return RAGPipeline()


def get_log_analyzer():
    """获取日志分析 Agent 单例"""
    from backend.agent.log_analyzer import LogAnalyzer
    return LogAnalyzer()
