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

    # ==================== 服务器配置 ====================
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ==================== LLM 配置（阿里云百炼 DashScope） ====================
    DASHSCOPE_API_KEY: str = ""           # 百炼 API Key
    DASHSCOPE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "deepseek-v3"         # 对话模型名称
    EMBEDDING_MODEL: str = "text-embedding-v3"  # 嵌入模型名称
    LLM_TEMPERATURE: float = 0.1           # 生成温度，越低越稳定
    LLM_MAX_TOKENS: int = 4096             # 最大输出 token 数

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
