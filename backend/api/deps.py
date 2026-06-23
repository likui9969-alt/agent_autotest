"""
依赖注入模块
提供 FastAPI 路由中使用的共享依赖（LLM 客户端、RAG 管线等）
各依赖采用 @lru_cache 实现应用级单例，避免每次请求重复初始化昂贵资源
（如 chromadb.PersistentClient、OpenAI 客户端等）。
"""
from functools import lru_cache


@lru_cache()
def get_settings():
    """获取应用配置（单例 — 委托给 settings 模块的缓存函数）"""
    from backend.config.settings import get_settings as _get_settings
    return _get_settings()


@lru_cache()
def get_llm_client():
    """获取 LLM 客户端单例（复用 OpenAI HTTP 连接池）"""
    from backend.llm.client import LLMClient
    return LLMClient()


@lru_cache()
def get_rag_pipeline():
    """获取 RAG 管线单例（复用 chromadb.PersistentClient 和嵌入模型连接）"""
    from backend.rag.pipeline import RAGPipeline
    return RAGPipeline()


@lru_cache()
def get_log_analyzer():
    """获取日志分析 Agent 单例（复用 LLM 客户端和 VectorStore 连接）"""
    from backend.agent.log_analyzer import LogAnalyzer
    return LogAnalyzer()


@lru_cache()
def get_test_executor():
    """获取测试执行 Agent 单例（复用日志分析 Agent）"""
    from backend.agent.test_executor import TestExecutorAgent
    return TestExecutorAgent(log_analyzer=get_log_analyzer())


@lru_cache()
def get_jira_creator():
    """获取 JIRA 创建 Agent 单例（复用 LLM 客户端）"""
    from backend.agent.jira_creator import JiraCreator
    return JiraCreator(llm_client=get_llm_client())
