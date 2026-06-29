"""
页面知识数据模型
================
存储网站页面探索结果，用于被测系统结构知识的复用。
"""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class PageElement(BaseModel):
    """页面元素摘要"""

    by: str = Field(default="", description="定位方式")
    value: str = Field(default="", description="定位值")
    type: str = Field(default="", description="元素类型")
    name: str = Field(default="", description="name 属性")
    id: str = Field(default="", description="id 属性")
    placeholder: str = Field(default="", description="placeholder")
    text: str = Field(default="", description="可见文本")
    href: str = Field(default="", description="链接地址")


class PageKnowledge(BaseModel):
    """页面知识条目"""

    id: str = Field(default="", description="唯一标识，通常使用 URL hash")
    url: str = Field(..., description="页面 URL")
    title: str = Field(default="", description="页面标题")
    page_type: str = Field(default="unknown", description="页面类型：login/search/order/unknown")
    inputs: list[PageElement] = Field(default_factory=list, description="输入框列表")
    buttons: list[PageElement] = Field(default_factory=list, description="按钮列表")
    links: list[PageElement] = Field(default_factory=list, description="链接列表")
    forms: list[dict] = Field(default_factory=list, description="表单结构")
    page_hash: str = Field(default="", description="页面内容哈希，用于变化检测")
    html_summary: str = Field(default="", description="页面文本摘要")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="创建时间")
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="更新时间")
    access_count: int = Field(default=1, description="复用次数")

    def summary_text(self) -> str:
        """生成可嵌入向量的文本摘要"""
        lines = [
            f"URL: {self.url}",
            f"标题: {self.title}",
            f"页面类型: {self.page_type}",
            f"输入框: {len(self.inputs)} 个",
            f"按钮: {len(self.buttons)} 个",
            f"链接: {len(self.links)} 个",
        ]
        if self.inputs:
            lines.append("输入框详情: " + ", ".join(
                f"{(i.name or i.id or i.placeholder)}({i.type})"
                for i in self.inputs
            ))
        if self.buttons:
            lines.append("按钮详情: " + ", ".join(b.text or "" for b in self.buttons))
        if self.html_summary:
            lines.append(f"摘要: {self.html_summary[:500]}")
        return "\n".join(lines)
