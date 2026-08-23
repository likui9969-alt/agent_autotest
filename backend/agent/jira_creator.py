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

    # 类级缓存：项目可用的 issue type 名称（避免每次建单都查 createmeta）
    _issue_type_cache: dict[str, str] = {}

    # issue type 候选优先级（兼容中英文模板的 team-managed / company-managed 项目）
    _ISSUE_TYPE_CANDIDATES = ("Bug", "故障", "任务", "Task", "Story", "故事")

    def _resolve_issue_type(self, client: "httpx.Client", jira_url: str, auth: tuple) -> str:
        """解析项目可用的缺陷类 issue type 名称

        team-managed（next-gen）项目可能没有 "Bug" 类型（如中文模板只有
        任务/故事/长篇故事），写死 "Bug" 会导致 400 错误。
        通过 createmeta 动态探测（需认证，未认证返回 404），
        按候选优先级选择，结果进程内缓存。
        """
        cached = JiraCreator._issue_type_cache.get(self.settings.JIRA_PROJECT_KEY)
        if cached:
            return cached

        default_type = "Bug"
        meta_url = (
            f"{jira_url}/rest/api/2/issue/createmeta/"
            f"{self.settings.JIRA_PROJECT_KEY}/issuetypes"
        )
        # 查询最多尝试 2 次（抵御偶发 SSL 握手超时/网络抖动）
        for attempt in range(2):
            try:
                resp = client.get(
                    meta_url, auth=auth, headers={"Content-Type": "application/json"}
                )
                if resp.status_code == 200:
                    names = [t.get("name", "") for t in resp.json().get("issueTypes", [])]
                    for candidate in self._ISSUE_TYPE_CANDIDATES:
                        if candidate in names:
                            JiraCreator._issue_type_cache[self.settings.JIRA_PROJECT_KEY] = candidate
                            logger.info(f"JIRA issue type 解析: 使用 '{candidate}'（项目可用: {names}）")
                            return candidate
                    # 无候选命中时取第一个非 subtask 类型兜底
                    non_subtask = [
                        t.get("name", "") for t in resp.json().get("issueTypes", [])
                        if not t.get("subtask", False)
                    ]
                    if non_subtask:
                        JiraCreator._issue_type_cache[self.settings.JIRA_PROJECT_KEY] = non_subtask[0]
                        return non_subtask[0]
                    return default_type
                logger.warning(
                    f"JIRA createmeta 返回 {resp.status_code}（尝试 {attempt + 1}/2）"
                )
            except Exception as e:
                logger.warning(
                    f"JIRA createmeta 查询失败（尝试 {attempt + 1}/2）: {e}"
                )
        return default_type

    def _call_jira_api(self, request: JiraCreateRequest) -> JiraCreateResponse:
        """调用 JIRA REST API 创建 Issue

        兼容性处理：
        1. issue type 动态解析（team-managed 项目可能没有 Bug 类型）
        2. priority/labels 字段被项目拒绝时自动剔除重试（team-managed
           项目可能未启用这些字段，400 errors 会指明字段名）
        """
        import httpx

        jira_url = self.settings.JIRA_URL.rstrip("/")
        api_url = f"{jira_url}/rest/api/2/issue"

        # 必填字段
        base_fields: dict = {
            "project": {"key": self.settings.JIRA_PROJECT_KEY},
            "summary": request.title,
            "description": request.description,
        }

        # 可选字段（team-managed 项目可能未启用，失败时剔除重试）
        optional_fields: dict = {
            "priority": {"name": request.priority or "Medium"},
            "labels": request.labels or ["ai-generated", "automation"],
        }

        # 如果有指派人（company-managed 项目用 name，team-managed 需要 accountId，
        # 此处保留原行为，指派失败会被下方容错逻辑剔除）
        if request.assignee:
            optional_fields["assignee"] = {"name": request.assignee}

        # HTTP Basic Auth
        auth = (self.settings.JIRA_USERNAME, self.settings.JIRA_API_TOKEN)

        # trust_env=False：不走系统代理环境变量，与 check_connection 行为保持一致
        with httpx.Client(timeout=30, trust_env=False) as client:
            # 动态解析 issue type（写入必填字段）
            issue_type = self._resolve_issue_type(client, jira_url, auth)
            base_fields["issuetype"] = {"name": issue_type}

            # 第一次尝试：带全部字段
            fields = {**base_fields, **optional_fields}
            response = client.post(
                api_url,
                json={"fields": fields},
                auth=auth,
                headers={"Content-Type": "application/json"},
            )

            # 400 且指明字段错误：剔除被拒的可选字段后重试一次
            if response.status_code == 400:
                try:
                    errors = response.json().get("errors", {})
                except Exception:
                    errors = {}
                rejected_optional = [k for k in optional_fields if k in errors]
                # issuetype 被拒说明动态解析也失败，直接报错（不能剔除必填字段）
                if rejected_optional and "issuetype" not in errors:
                    logger.warning(
                        f"JIRA 项目拒绝字段 {rejected_optional}（可能未启用），剔除后重试"
                    )
                    retry_fields = {**base_fields, **{
                        k: v for k, v in optional_fields.items()
                        if k not in rejected_optional
                    }}
                    response = client.post(
                        api_url,
                        json={"fields": retry_fields},
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
