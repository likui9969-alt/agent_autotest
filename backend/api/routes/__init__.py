"""
路由注册汇总模块
将各业务路由子模块统一注册到 FastAPI 应用上
"""
import logging
from fastapi import FastAPI

logger = logging.getLogger("ai_rd_agent")


def register_routes(app: FastAPI):
    """注册所有业务路由到应用实例"""
    # 各路由前缀定义
    route_modules = [
        ("health", "health", ""),                   # 健康检查（路由自带 /health 路径）
        ("knowledge", "knowledge", "/api/v1/knowledge"),  # 知识库管理
        ("rag", "rag", "/api/v1/rag"),            # 智能问答
        ("analysis", "analysis", "/api/v1/analysis"),  # 日志分析
        ("testing", "testing", "/api/v1/testing"),  # 自动化测试
        ("jira", "jira", "/api/v1/jira"),         # JIRA 集成
        ("agents", "agents", "/api/v1/agent"),    # Agent 执行（LangGraph + ReAct）
    ]

    for module_name, route_name, prefix in route_modules:
        try:
            module = __import__(
                f"backend.api.routes.{module_name}",
                fromlist=[route_name],
            )
            router = getattr(module, "router", None)
            if router:
                app.include_router(router, prefix=prefix)
                logger.info(f"   [OK] 路由已注册: {prefix or '/'}")
            else:
                logger.warning(f"   ⚠ 路由模块 {module_name} 缺少 router 对象")
        except ImportError:
            logger.info(f"   ⊘ 路由模块 {module_name} 尚未实现，已跳过")
