"""
自动化测试相关数据模型
包含测试执行请求、测试结果、测试报告等
"""
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    """测试执行状态"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestScenario(str, Enum):
    """预定义的测试场景"""
    LOGIN = "login"       # 登录流程测试
    SEARCH = "search"     # 搜索流程测试
    ORDER = "order"       # 下单流程测试


# ==================== 请求模型 ====================

class TestRunRequest(BaseModel):
    """测试执行请求"""
    scenarios: list[TestScenario] = Field(
        default=[TestScenario.LOGIN],
        description="要执行的测试场景列表"
    )
    base_url: str = Field(default="https://example.com", description="被测网站地址")
    headless: bool = Field(default=True, description="是否使用无头浏览器模式")
    timeout_seconds: int = Field(default=30, description="单个测试步骤超时时间（秒）")
    auto_analyze: bool = Field(
        default=True,
        description="测试失败时是否自动调用日志分析 Agent"
    )
    sandbox: bool = Field(
        default=False,
        description="沙盒模式：模拟 Selenium 执行（无需 Chrome，用于演示）"
    )


# ==================== 测试步骤结果 ====================

class TestStepResult(BaseModel):
    """单个测试步骤的执行结果"""
    step_name: str = Field(..., description="步骤名称")
    status: TestStatus = Field(..., description="步骤执行状态")
    duration_ms: float = Field(default=0, description="步骤耗时（毫秒）")
    error_message: str = Field(default="", description="错误信息（如有）")
    screenshot_path: str = Field(default="", description="截图保存路径（如有）")


# ==================== 测试用例结果 ====================

class TestCaseResult(BaseModel):
    """单个测试用例的执行结果"""
    scenario: str = Field(..., description="测试场景名称")
    status: TestStatus = Field(..., description="用例执行状态")
    start_time: datetime = Field(default_factory=datetime.now, description="开始时间")
    end_time: datetime | None = Field(default=None, description="结束时间")
    duration_ms: float = Field(default=0, description="总耗时（毫秒）")
    steps: list[TestStepResult] = Field(default_factory=list, description="各步骤结果")
    error_message: str = Field(default="", description="整体错误信息")
    selenium_logs: str = Field(default="", description="Selenium 执行日志")


# ==================== 测试报告 ====================

class TestReport(BaseModel):
    """完整的测试执行报告"""
    report_id: str = Field(default="", description="报告唯一 ID")
    base_url: str = Field(default="", description="被测网站地址")
    executed_at: datetime = Field(default_factory=datetime.now, description="执行时间")

    # 统计信息
    total_scenarios: int = Field(default=0, description="执行场景总数")
    passed_count: int = Field(default=0, description="通过数")
    failed_count: int = Field(default=0, description="失败数")
    pass_rate: float = Field(default=0.0, description="通过率（0~1）")

    # 各场景详情
    results: list[TestCaseResult] = Field(default_factory=list, description="各场景执行结果")

    # 失败分析（如有）
    failure_analysis: list[str] = Field(
        default_factory=list,
        description="对失败用例的 AI 分析结果"
    )
