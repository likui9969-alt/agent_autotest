"""
JIRA 缺陷创建 Agent 模块
根据故障分析结果自动生成并创建 JIRA 缺陷单
"""
import logging
from backend.llm.client import LLMClient
from backend.llm.prompts import get_template
from backend.models.jira import JiraCreateRequest, JiraCreateResponse
from backend.config.settings import get_settings

logger = logging.getLogger("ai_rd_agent")


class JiraCreator:
    """JIRA 缺陷创建 Agent

    使用示例：
        creator = JiraCreator()
        response = creator.create_issue(JiraCreateRequest(
            title="登录页面加载超时",
            description="用户反馈登录页面加载超过30秒",
            log_content="...",
            ai_analysis="可能原因：网络延迟...",
        ))
    """

    def __init__(self, llm_client: LLMClient | None = None):
        if llm_client:
            self.llm_client = llm_client
        else:
            from backend.api.deps import get_llm_client
            self.llm_client = get_llm_client()
        self.settings = get_settings()
        logger.info("JIRA 创建 Agent 已初始化")

    def create_issue(self, request: JiraCreateRequest) -> JiraCreateResponse:
        """创建 JIRA 缺陷单

        Args:
            request: 缺陷创建请求（含标题、描述、日志、AI分析等）

        Returns:
            创建结果（含 Issue Key 和链接）
        """
        logger.info(f"开始创建 JIRA 缺陷: {request.title}")

        # ---- 步骤 1：用 LLM 优化缺陷描述 ----
        if request.ai_analysis:
            refined = self._refine_with_llm(request)
            if refined:
                request.description = refined

        # ---- 步骤 2：调用 JIRA API 创建 Issue ----
        if not self.settings.JIRA_URL:
            # JIRA 未配置时明确跳过，避免误导 Agent
            logger.warning("JIRA 未配置，跳过创建")
            return JiraCreateResponse(
                status="skipped",
                issue_key="",
                issue_url="",
                message="JIRA 未配置（缺少 JIRA_URL），无法创建缺陷单。请在 .env 中配置 JIRA_URL、JIRA_USERNAME、JIRA_API_TOKEN、JIRA_PROJECT_KEY。",
            )

        try:
            response = self._call_jira_api(request)
            return response
        except Exception as e:
            logger.error(f"JIRA API 调用失败: {e}", exc_info=True)
            return JiraCreateResponse(
                status="failed",
                message=f"JIRA 创建失败: {str(e)}",
            )

    def _refine_with_llm(self, request: JiraCreateRequest) -> str | None:
        """使用 LLM 优化缺陷描述"""
        try:
            template = get_template("jira_creation")
            messages = [
                {"role": "system", "content": template.system},
                {"role": "user", "content": template.user.format(
                    summary=request.title,
                    analysis=request.ai_analysis or "无",
                    log_snippet=request.log_content[:2000] or "无",
                )},
            ]
            response = self.llm_client.chat(
                messages=messages,
                temperature=template.temperature,
            )
            # 提取 LLM 返回的标题和描述
            return response
        except Exception as e:
            logger.warning(f"LLM 优化缺陷描述失败: {e}")
            return None

    def check_connection(self) -> dict:
        """检查 JIRA 连接状态

        Returns:
            {"status": "connected" | "unconfigured" | "failed", "message": str}
        """
        import httpx

        if not self.settings.JIRA_URL:
            return {
                "status": "unconfigured",
                "message": "JIRA 未配置（缺少 JIRA_URL）",
            }

        jira_url = self.settings.JIRA_URL.rstrip("/")
        check_url = f"{jira_url}/rest/api/2/serverInfo"
        auth = (self.settings.JIRA_USERNAME, self.settings.JIRA_API_TOKEN)

        try:
            with httpx.Client(trust_env=False, timeout=10) as client:
                response = client.get(check_url, auth=auth, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "connected",
                    "message": f"连接成功（JIRA {data.get('version', 'unknown')}）",
                    "base_url": jira_url,
                    "version": data.get("version", ""),
                }
            else:
                return {
                    "status": "failed",
                    "message": f"JIRA 返回错误 {response.status_code}: {response.text[:200]}",
                }
        except Exception as e:
            return {
                "status": "failed",
                "message": f"连接失败: {str(e)}",
            }

    def _call_jira_api(self, request: JiraCreateRequest) -> JiraCreateResponse:
        """调用 JIRA REST API 创建 Issue"""
        import httpx

        jira_url = self.settings.JIRA_URL.rstrip("/")
        api_url = f"{jira_url}/rest/api/2/issue"

        # 构建 JIRA API 请求体
        issue_data = {
            "fields": {
                "project": {"key": self.settings.JIRA_PROJECT_KEY},
                "summary": request.title,
                "description": request.description,
                "issuetype": {"name": "Bug"},
                "priority": {"name": request.priority or "Medium"},
                "labels": request.labels or ["ai-generated", "automation"],
            }
        }

        # 如果有指派人
        if request.assignee:
            issue_data["fields"]["assignee"] = {"name": request.assignee}

        # HTTP Basic Auth
        auth = (self.settings.JIRA_USERNAME, self.settings.JIRA_API_TOKEN)

        with httpx.Client(timeout=30) as client:
            response = client.post(
                api_url,
                json=issue_data,
                auth=auth,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code in (200, 201):
            data = response.json()
            issue_key = data.get("key", "UNKNOWN")
            return JiraCreateResponse(
                status="success",
                issue_key=issue_key,
                issue_url=f"{jira_url}/browse/{issue_key}",
                message="缺陷单创建成功",
            )
        else:
            error_msg = response.text[:500]
            logger.error(f"JIRA API 返回错误 {response.status_code}: {error_msg}")
            return JiraCreateResponse(
                status="failed",
                message=f"JIRA API 错误 ({response.status_code}): {error_msg}",
            )
