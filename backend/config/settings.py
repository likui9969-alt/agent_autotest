"""
应用配置管理模块
使用 Pydantic BaseSettings 读取 .env 文件，提供类型安全的配置对象
"""
import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


# 项目根目录 — 从当前文件向上查找 3 级 (config -> backend -> agentone_test)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """应用全局配置，所有值从 .env 文件或环境变量加载"""

    # ==================== 应用基础配置 ====================
    APP_NAME: str = "AI研发效能智能体"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ALLOWED_ORIGINS: list[str] = ["*"]
    # CORS 允许的源，留空或 ["*"] 表示允许所有。生产环境应设为具体域名。
    API_TOKEN: str = ""
    # API 访问 Token。配置后所有 API 请求需携带 Authorization: Bearer <token>。
    # 留空时不启用认证（向后兼容）。

    # ==================== 日志配置 ====================
    LOG_FORMAT: str = "text"
    # 日志输出格式: "text"（人类可读）或 "json"（Logstash 兼容 JSON，推荐容器生产环境使用）

    # ==================== 速率限制配置 ====================
    RATE_LIMIT_ENABLED: bool = True
    # 是否启用速率限制。生产环境建议启用，开发调试可关闭。
    RATE_LIMIT_DEFAULT: str = "30/minute"
    # 默认速率限制（每 IP），格式: "<次数>/<时间单位>"。支持 second/minute/hour/day。
    # P0 关键端点和 Agent 端点使用独立限制（见下方）。

    # ==================== Agent 执行配置 ====================
    AGENT_TIMEOUT_SECONDS: int = 120
    # Agent 单次执行超时时间（秒）。超过该时间未完成则强制终止。
    AGENT_RECURSION_LIMIT: int = 50
    # LangGraph recursion_limit，控制图遍历最大深度。AGENT_TIMEOUT_SECONDS * 2 为经验参考值。

    # ==================== LLM 重试配置 ====================
    LLM_RETRY_MAX_ATTEMPTS: int = 3
    LLM_RETRY_BASE_DELAY: float = 1.0
    LLM_RETRY_MAX_DELAY: float = 8.0
    # LLM 调用失败时指数退避重试参数：delay = base_delay * 2^attempt + jitter

    # ==================== LLM 熔断器配置 ====================
    LLM_CIRCUIT_BREAKER_THRESHOLD: int = 5
    # 连续失败 N 次后熔断器开启（后续请求快速失败，不再调用 LLM）
    LLM_CIRCUIT_BREAKER_RECOVERY: float = 30.0
    # 熔断开启后等待多少秒进入 HALF_OPEN 探测状态
    LLM_CIRCUIT_BREAKER_SUCCESS: int = 2
    # HALF_OPEN 下连续成功 N 次后完全恢复

    # ==================== 服务器配置 ====================
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ==================== LLM 主 Provider 配置 ====================
    LLM_PROVIDER: str = "dashscope"
    # 默认 LLM Provider: dashscope / openai / ollama

    # ==================== LLM 配置（阿里云百炼 DashScope） ====================
    DASHSCOPE_API_KEY: str = ""           # 百炼 API Key
    DASHSCOPE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "deepseek-v3"         # 对话模型名称
    EMBEDDING_MODEL: str = "text-embedding-v3"  # 嵌入模型名称
    LLM_TEMPERATURE: float = 0.1           # 生成温度，越低越稳定
    LLM_MAX_TOKENS: int = 4096             # 最大输出 token 数

    # ==================== LLM 配置（OpenAI 官方） ====================
    OPENAI_API_KEY: str = ""              # OpenAI API Key
    OPENAI_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"     # OpenAI 对话模型名称
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"  # OpenAI 嵌入模型名称

    # ==================== LLM 回退（Fallback）配置 ====================
    LLM_FALLBACK_ENABLED: bool = False
    # 是否启用 LLM 回退。启用后当主 LLM 不可用时自动切换到备用 Provider。
    LLM_PROVIDERS: list[str] = ["dashscope"]
    # provider 优先级列表。默认只走 dashscope。设置 ["dashscope", "ollama"] 启用回退。
    OLLAMA_URL: str = ""
    # Ollama 服务地址，如 "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    # Ollama 对话模型名称
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    # Ollama 嵌入模型名称

    # ==================== RAG 配置 ====================
    CHUNK_SIZE: int = 1000                 # 文档切分块大小
    CHUNK_OVERLAP: int = 200               # 文档切分块重叠量
    RETRIEVER_TOP_K: int = 5               # 检索返回的最相关文档数
    CHROMA_PERSIST_DIR: str = ""           # Chroma 持久化目录（留空则自动设为 data/chroma）

    # ==================== JIRA 配置（可选） ====================
    JIRA_URL: str = ""                     # JIRA 实例地址
    JIRA_USERNAME: str = ""                # JIRA 用户名
    JIRA_API_TOKEN: str = ""               # JIRA API Token
    JIRA_PROJECT_KEY: str = ""             # JIRA 项目 Key

    # ==================== Selenium 配置 ====================
    CHROMEDRIVER_PATH: str = ""           # chromedriver 路径（留空则自动检测项目根目录 / PATH）
    CHROME_BINARY_PATH: str = ""          # Chrome 浏览器可执行文件路径（留空则使用系统默认）

    # ==================== 路径配置 ====================
    DATA_DIR: str = ""                     # 数据存储根目录（留空则自动设为 data/）
    UPLOAD_DIR: str = ""                   # 文档上传目录（留空则自动设为 data/docs/）
    LOG_DIR: str = ""                      # 日志存储目录（留空则自动设为 data/logs/）
    BACKUP_DIR: str = ""                   # Chroma 备份存储目录（留空则自动设为 data/backups/）

    def get_chroma_dir(self) -> str:
        """获取 Chroma 持久化目录的绝对路径"""
        if self.CHROMA_PERSIST_DIR:
            return self.CHROMA_PERSIST_DIR
        return str(PROJECT_ROOT / "data" / "chroma")

    def get_upload_dir(self) -> str:
        """获取文档上传目录的绝对路径"""
        if self.UPLOAD_DIR:
            return self.UPLOAD_DIR
        return str(PROJECT_ROOT / "data" / "docs")

    def get_log_dir(self) -> str:
        """获取日志存储目录的绝对路径"""
        if self.LOG_DIR:
            return self.LOG_DIR
        return str(PROJECT_ROOT / "data" / "logs")

    def get_backup_dir(self) -> str:
        """获取 Chroma 备份存储目录的绝对路径"""
        if self.BACKUP_DIR:
            return self.BACKUP_DIR
        backup = PROJECT_ROOT / "data" / "backups"
        backup.mkdir(parents=True, exist_ok=True)
        return str(backup)

    def get_chromedriver_path(self) -> str:
        """获取 chromedriver 可执行文件路径

        优先级：显式配置 > 项目根目录下的 chromedriver.exe > 系统 PATH（返回空串）
        """
        if self.CHROMEDRIVER_PATH:
            return self.CHROMEDRIVER_PATH
        # 自动检测项目根目录下的 chromedriver.exe / chromedriver
        for name in ("chromedriver.exe", "chromedriver"):
            candidate = PROJECT_ROOT / name
            if candidate.exists():
                return str(candidate)
        return ""

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略 .env 中未定义的字段


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例（带缓存，避免重复读取 .env）"""
    return Settings()
