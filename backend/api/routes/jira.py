"""
JIRA 集成路由
POST /api/v1/jira/create — 自动创建 JIRA 缺陷单
"""
import logging
from fastapi import APIRouter

from backend.models.jira import JiraCreateRequest, JiraCreateResponse
from backend.agent.jira_creator import JiraCreator

logger = logging.getLogger("ai_rd_agent")
router = APIRouter(tags=["JIRA集成"])


@router.post("/create", response_model=JiraCreateResponse)
async def create_jira_issue(request: JiraCreateRequest):
    """自动创建 JIRA 缺陷单

    基于故障分析结果生成缺陷描述，调用 JIRA API 创建 Issue。

    请求示例：
    {
        "title": "登录页面加载超时（>30s）",
        "description": "用户在点击登录按钮后，页面加载超过30秒无响应",
        "priority": "High",
        "log_content": "TimeoutException: Page load timeout...",
        "ai_analysis": "可能原因：后端接口响应慢，数据库连接池耗尽",
        "labels": ["bug", "automation", "login"],
        "assignee": "zhangsan"
    }
    """
    creator = JiraCreator()
    response = creator.create_issue(request)

    if response.status == "success":
        logger.info(f"JIRA 缺陷已创建: {response.issue_key}")
    else:
        logger.warning(f"JIRA 创建失败: {response.message}")

    return response
