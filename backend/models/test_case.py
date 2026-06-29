"""
测试用例数据模型
================
定义测试用例的结构化模型，用于自动生成、存储和导出。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """测试用例模型"""

    id: str = Field(default="", description="用例唯一标识")
    module: str = Field(default="", description="所属模块")
    title: str = Field(..., description="用例标题")
    objective: str = Field(default="", description="测试目标")
    preconditions: str = Field(default="", description="前置条件")
    steps: list[str] = Field(default_factory=list, description="测试步骤")
    expected_result: str = Field(default="", description="预期结果")
    priority: str = Field(default="中", description="优先级（高/中/低）")
    source: str = Field(default="", description="来源需求或文档")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="创建时间")

    def to_row(self) -> dict[str, Any]:
        """导出为平铺字典（用于 CSV/Excel）"""
        return {
            "模块": self.module,
            "用例标题": self.title,
            "测试目标": self.objective,
            "前置条件": self.preconditions,
            "测试步骤": "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.steps)),
            "预期结果": self.expected_result,
            "优先级": self.priority,
            "来源": self.source,
            "创建时间": self.created_at,
        }


class TestCaseGenerateRequest(BaseModel):
    """测试用例生成请求"""

    requirement: str = Field(..., description="自然语言需求描述", min_length=5)
    scenario: str = Field(default="", description="补充业务场景说明")
    context: str = Field(default="", description="已上传的需求文档内容")
    use_knowledge: bool = Field(default=True, description="是否检索知识库补充历史用例")
    count: int = Field(default=5, ge=1, le=20, description="期望生成用例数量")


class TestCaseGenerateResponse(BaseModel):
    """测试用例生成响应"""

    status: str = Field(default="success", description="生成状态")
    generated_count: int = Field(default=0, description="生成用例数量")
    test_cases: list[TestCase] = Field(default_factory=list, description="生成的用例列表")
    message: str = Field(default="", description="状态说明")
