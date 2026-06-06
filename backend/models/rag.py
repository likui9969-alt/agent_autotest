"""
RAG 检索问答相关数据模型
包含查询请求、检索结果、引用来源等
"""
from datetime import datetime
from pydantic import BaseModel, Field


# ==================== 请求模型 ====================

class RAGQueryRequest(BaseModel):
    """RAG 问答请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20, description="检索返回的最相关文档数")
    search_type: str = Field(
        default="similarity",
        description="检索方式：similarity（相似度） / mmr（最大边际相关性）"
    )
    include_sources: bool = Field(default=True, description="是否在回答中附带引用来源")


# ==================== 检索结果模型 ====================

class RetrievedDocument(BaseModel):
    """检索到的单个文档块"""
    chunk_id: str = Field(..., description="文档块唯一 ID")
    content: str = Field(..., description="文档块文本内容")
    source_file: str = Field(default="", description="来源文件名")
    score: float = Field(default=0.0, description="相似度分数（0~1）")
    metadata: dict = Field(default_factory=dict, description="附加元数据")


# ==================== 响应模型 ====================

class SourceCitation(BaseModel):
    """回答引用来源"""
    source_file: str = Field(..., description="来源文档名")
    chunk_index: int = Field(default=0, description="文档块索引")
    excerpt: str = Field(default="", description="引用摘要（截取前 200 字）")
    score: float = Field(default=0.0, description="相关度分数")


class RAGQueryResponse(BaseModel):
    """RAG 问答响应"""
    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="RAG 生成的回答")
    sources: list[SourceCitation] = Field(default_factory=list, description="引用来源列表")
    retrieved_count: int = Field(default=0, description="实际检索到的文档数")
    response_time_ms: float = Field(default=0, description="响应耗时（毫秒）")
    answered_at: datetime = Field(default_factory=datetime.now, description="回答时间")
