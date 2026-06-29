"""
测试用例自动生成器
==================
基于 LLM 与 RAG 知识库，根据自然语言需求自动生成结构化测试用例。
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Optional

from backend.llm.client import LLMClient
from backend.llm.prompts import get_template
from backend.models.test_case import TestCase, TestCaseGenerateRequest

logger = logging.getLogger("ai_rd_agent")


_GENERATION_SYSTEM_PROMPT = """你是一个资深测试用例设计专家。
请根据用户提供的需求描述和补充上下文，生成结构化的测试用例。

输出必须满足以下要求：
1. 返回 JSON 数组，每个元素是一个用例对象
2. 每个用例对象包含字段：module（模块）、title（标题）、objective（测试目标）、preconditions（前置条件，字符串）、steps（步骤，字符串数组）、expected_result（预期结果）、priority（优先级：高/中/低）
3. 用例应覆盖正常流程、异常流程和边界情况
4. 优先使用中文
5. 不要输出 JSON 以外的任何解释文字"""


class TestCaseGenerator:
    """测试用例生成器

    使用示例：
        generator = TestCaseGenerator()
        response = generator.generate(TestCaseGenerateRequest(
            requirement="用户登录功能",
            scenario="电商 App 登录页",
            context="",
        ))
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        if llm_client:
            self.llm_client = llm_client
        else:
            from backend.api.deps import get_llm_client
            self.llm_client = get_llm_client()
        logger.info("测试用例生成器已初始化")

    def generate(self, request: TestCaseGenerateRequest) -> list[TestCase]:
        """根据需求生成测试用例"""
        knowledge_context = ""
        if request.use_knowledge:
            try:
                knowledge_context = self._retrieve_knowledge(request.requirement)
            except Exception as e:
                logger.warning(f"检索知识库失败，继续基于需求生成: {e}")

        combined_context = request.context or ""
        if knowledge_context:
            combined_context = f"历史相关用例/案例：\n{knowledge_context}\n\n{combined_context}"

        prompt = self._build_prompt(request, combined_context)
        raw_output = self.llm_client.chat(
            messages=[
                {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )

        cases = self._parse_cases(raw_output)

        # 补充元数据
        now = datetime.now().isoformat()
        for case in cases:
            case.id = str(uuid.uuid4())[:8]
            case.source = request.requirement[:200]
            case.created_at = now

        # 限制返回数量
        if len(cases) > request.count:
            cases = cases[:request.count]

        logger.info(f"测试用例生成完成: {len(cases)} 条")
        return cases

    def _retrieve_knowledge(self, query: str) -> str:
        """从 RAG 知识库检索相关历史用例"""
        from backend.api.deps import get_rag_pipeline

        pipeline = get_rag_pipeline()
        docs = pipeline.retriever.similarity_search(query, top_k=3)
        if not docs:
            return ""

        parts = []
        for doc in docs:
            source = doc.metadata.get("filename", "未知")
            parts.append(f"来源 {source}:\n{doc.page_content[:800]}")
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, request: TestCaseGenerateRequest, context: str) -> str:
        """构建生成提示词"""
        template = get_template("test_generation")
        scenario_text = request.scenario or request.requirement
        return template.user.format(
            scenario=scenario_text,
            context=context or "无",
        )

    def _parse_cases(self, raw: str) -> list[TestCase]:
        """解析 LLM 输出为 TestCase 列表"""
        # 尝试提取 JSON 代码块
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if code_block:
            raw = code_block.group(1).strip()

        # 尝试定位 JSON 数组
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            # 可能是单个 JSON 对象
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = f"[{raw[start:end+1]}]"
            else:
                logger.warning(f"无法解析 LLM 输出为 JSON: {raw[:200]}")
                return []
        else:
            raw = raw[start:end+1]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}\n输出: {raw[:500]}")
            return []

        cases: list[TestCase] = []
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                continue
            try:
                cases.append(TestCase(
                    module=item.get("module", ""),
                    title=item.get("title", ""),
                    objective=item.get("objective", ""),
                    preconditions=item.get("preconditions", ""),
                    steps=item.get("steps", []),
                    expected_result=item.get("expected_result", ""),
                    priority=item.get("priority", "中"),
                ))
            except Exception as e:
                logger.warning(f"单条用例解析失败: {e}")
                continue

        return cases
