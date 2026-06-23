"""
日志分析相关数据模型
包含日志上传、分析请求/响应、异常模式等
"""
from datetime import datetime
from pydantic import BaseModel, Field


# ==================== 请求模型 ====================

class LogAnalysisRequest(BaseModel):
    """日志分析请求"""
    log_content: str = Field(..., description="日志文本内容（支持直接粘贴或文件内容）")
    filename: str = Field(default="unknown.log", description="日志文件名（用于标识）")
    include_historical: bool = Field(default=True, description="是否检索历史相似案例")
    top_k: int = Field(default=3, description="检索历史案例数量")


# ==================== 异常信息模型 ====================

class ExceptionInfo(BaseModel):
    """从日志中提取的异常信息"""
    exception_type: str = Field(..., description="异常类型（TimeoutException 等）")
    message: str = Field(default="", description="异常消息")
    traceback_lines: list[str] = Field(default_factory=list, description="Traceback 关键行")
    file_path: str = Field(default="", description="发生异常的文件路径")
    line_number: int | None = Field(default=None, description="发生异常的行号")


# ==================== 历史案例模型 ====================

class HistoricalCase(BaseModel):
    """历史相似故障案例"""
    case_id: str = Field(default="", description="案例标识")
    title: str = Field(default="", description="案例标题")
    similarity_score: float = Field(..., description="相似度分数（0~1）")
    description: str = Field(default="", description="案例描述")
    root_cause: str = Field(default="", description="根因")
    solution: str = Field(default="", description="修复方案")
    source: str = Field(default="knowledge_base", description="案例来源")


# ==================== 分析结果模型 ====================

class AnalysisResult(BaseModel):
    """完整的故障分析报告"""
    # 基本信息
    analysis_id: str = Field(default="", description="分析任务唯一 ID")
    filename: str = Field(default="", description="被分析的日志文件名")
    analyzed_at: datetime = Field(default_factory=datetime.now, description="分析时间")

    # 问题摘要
    summary: str = Field(default="", description="问题摘要（一句话描述）")

    # 异常详情
    exceptions_found: list[ExceptionInfo] = Field(default_factory=list, description="检测到的异常列表")

    # 可能原因
    possible_causes: list[str] = Field(default_factory=list, description="可能的根本原因列表")

    # 历史案例
    historical_cases: list[HistoricalCase] = Field(default_factory=list, description="相关联的历史案例")

    # 修复建议
    fix_suggestions: list[str] = Field(default_factory=list, description="修复建议列表")

    # 严重等级
    severity: str = Field(default="中", description="严重等级：高 / 中 / 低")

    # 原始 AI 分析全文（用于展示完整分析过程）
    raw_analysis: str = Field(default="", description="AI 原始分析全文")


# ==================== 异常类型枚举 ====================

KNOWN_EXCEPTIONS = [
    "TimeoutException",
    "NoSuchElementException",
    "AssertionError",
    "ConnectionError",
    "SQLException",
    "OperationalError",
    "IntegrityError",
    "ProgrammingError",
    "AttributeError",
    "KeyError",
    "ValueError",
    "HTTPError",
    "AuthError",
    "AuthenticationError",
]
