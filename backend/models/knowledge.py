"""
知识库管理相关数据模型
包含文档上传、知识库状态查询等请求/响应模型
"""
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """文档上传成功后的响应"""
    filename: str = Field(..., description="原始文件名")
    file_type: str = Field(..., description="文件类型（txt / pdf / docx）")
    file_size_bytes: int = Field(..., description="文件大小（字节）")
    chunk_count: int = Field(..., description="切分后的文本块数量")
    status: str = Field(default="success", description="处理状态")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="上传时间")


class KnowledgeBaseStats(BaseModel):
    """知识库统计信息"""
    total_documents: int = Field(..., description="已上传文档总数")
    total_chunks: int = Field(..., description="向量库中块总数")
    collection_name: str = Field(default="knowledge_base", description="Chroma 集合名称")
    persist_directory: str = Field(default="", description="持久化目录路径")
    last_updated: datetime | None = Field(default=None, description="最近更新时间")


class RebuildResponse(BaseModel):
    """重建向量库的响应"""
    status: str = Field(..., description="操作状态")
    documents_processed: int = Field(default=0, description="处理的文档数")
    chunks_created: int = Field(default=0, description="创建的块数")
    message: str = Field(default="", description="操作结果描述")
