"""
JIRA 集成相关数据模型
包含缺陷创建请求/响应模型
"""
from pydantic import BaseModel, Field


class JiraCreateRequest(BaseModel):
    """JIRA 缺陷创建请求"""
    title: str = Field(..., description="缺陷标题（简洁明了）", min_length=5, max_length=200)
    description: str = Field(..., description="缺陷描述（含复现步骤、影响范围等）", min_length=10)
    priority: str = Field(
        default="Medium",
        description="优先级：Highest / High / Medium / Low"
    )
    log_content: str = Field(default="", description="相关日志内容")
    ai_analysis: str = Field(default="", description="AI 分析结果")
    labels: list[str] = Field(default_factory=list, description="标签列表（如 ['bug', 'automation']）")
    assignee: str = Field(default="", description="指派人用户名")


class JiraCreateResponse(BaseModel):
    """JIRA 缺陷创建响应"""
    status: str = Field(..., description="创建状态（success / failed）")
    issue_key: str = Field(default="", description="JIRA Issue Key（如 PROJ-123）")
    issue_url: str = Field(default="", description="JIRA Issue 链接")
    message: str = Field(default="", description="结果描述信息")
