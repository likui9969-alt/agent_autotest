"""
Prompt 模板集中管理模块
所有 LLM 交互的 Prompt 模板统一在此定义，便于维护和迭代
"""
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Prompt 模板数据结构"""
    name: str           # 模板名称
    system: str         # 系统角色设定
    user: str           # 用户消息模板（支持 {variable} 占位符）
    temperature: float  # 推荐温度


# ==================== RAG 问答模板 ====================

RAG_QUERY_PROMPT = PromptTemplate(
    name="RAG问答",
    system="""你是一个专业的测试工程师助手，擅长分析软件测试中的各种问题。
请根据提供的知识库内容回答用户问题。

要求：
1. 如果知识库中有相关案例，请引用并给出具体原因和解决方案
2. 如果知识库中没有直接相关信息，请基于你的专业知识给出建议
3. 回答要结构清晰，包含：原因分析、解决方案、相关日志片段（如有）
4. 使用中文回答""",
    user="""用户问题：{question}

相关知识库内容：
{context}

请基于以上信息回答问题。""",
    temperature=0.1,
)


# ==================== 日志分析模板 ====================

LOG_ANALYSIS_PROMPT = PromptTemplate(
    name="测试日志分析",
    system="""你是一个资深的软件测试故障分析专家，擅长从测试日志中定位问题根因。

你的分析能力覆盖以下异常类型：
- TimeoutException：操作超时，可能是网络延迟、服务器响应慢或元素加载超时
- NoSuchElementException：元素未找到，可能是页面结构变化、定位器失效或页面未加载完成
- AssertionError：断言失败，预期的结果与实际结果不匹配
- ConnectionError：连接错误，可能是网络不通、服务未启动或防火墙拦截
- SQL异常：数据库操作失败，可能是语法错误、连接池耗尽或数据约束冲突

分析步骤：
1. 识别 Traceback 和异常类型
2. 提取关键错误信息
3. 推断可能的根本原因
4. 关联历史相似案例
5. 给出修复建议""",
    user="""请分析以下测试日志，并生成故障分析报告。

日志内容：
```
{log_content}
```

相关历史案例（知识库检索结果）：
```
{historical_cases}
```

请按以下格式输出分析报告：
### 问题摘要
### 可能原因
### 历史案例
### 修复建议""",
    temperature=0.1,
)


# ==================== 测试用例生成模板 ====================

TEST_GENERATION_PROMPT = PromptTemplate(
    name="测试用例生成",
    system="""你是一个测试用例设计专家，能够根据业务描述自动生成详细的测试用例。
生成的测试用例应包含：测试目标、前置条件、测试步骤、预期结果、优先级""",
    user="""请为以下业务场景生成测试用例：

场景描述：{scenario}
相关需求文档：{context}

请生成包含以下内容的测试用例：
1. 用例名称
2. 测试目标
3. 前置条件
4. 测试步骤（分步骤描述）
5. 预期结果
6. 优先级（高/中/低）""",
    temperature=0.3,
)


# ==================== JIRA 缺陷单模板 ====================

JIRA_CREATION_PROMPT = PromptTemplate(
    name="JIRA缺陷单生成",
    system="""你是一个善于归纳总结的测试工程师，能够将故障分析结果转化为清晰、可操作的 JIRA 缺陷描述。""",
    user="""请根据以下故障信息，生成 JIRA 缺陷单的标题和描述。

故障摘要：{summary}
分析结果：{analysis}
日志片段：{log_snippet}

生成的标题应简洁明了，描述应包含：
1. 问题现象
2. 复现步骤（如可推断）
3. 影响范围
4. AI 分析建议""",
    temperature=0.2,
)


# ==================== 模板注册表 ====================

# 所有可用模板的字典，按名称索引
TEMPLATES = {
    "rag_query": RAG_QUERY_PROMPT,
    "log_analysis": LOG_ANALYSIS_PROMPT,
    "test_generation": TEST_GENERATION_PROMPT,
    "jira_creation": JIRA_CREATION_PROMPT,
}


def get_template(name: str) -> PromptTemplate:
    """获取指定名称的 Prompt 模板

    Args:
        name: 模板名称（rag_query / log_analysis / test_generation / jira_creation）

    Returns:
        PromptTemplate 实例

    Raises:
        KeyError: 模板名称不存在时抛出
    """
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise KeyError(f"模板 '{name}' 不存在。可用模板: {available}")
    return TEMPLATES[name]
