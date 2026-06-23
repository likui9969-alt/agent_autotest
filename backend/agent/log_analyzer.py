"""
日志分析 Agent 模块
自动识别日志中的异常信息，检索历史案例，生成故障分析报告

分析流程：
1. 识别 Traceback 和异常类型
2. 提取关键错误信息
3. 在知识库中检索历史相似案例
4. 调用 LLM 生成综合分析报告
5. 返回结构化分析结果
"""
import re
import logging
import uuid

from backend.llm.client import LLMClient
from backend.llm.prompts import get_template
from backend.rag.retriever import Retriever
from backend.rag.embeddings import EmbeddingGenerator
from backend.rag.vector_store import VectorStore
from backend.models.analysis import (
    LogAnalysisRequest,
    AnalysisResult,
    ExceptionInfo,
    HistoricalCase,
    KNOWN_EXCEPTIONS,
)

logger = logging.getLogger("ai_rd_agent")


class LogAnalyzer:
    """测试日志分析 Agent

    使用示例：
        analyzer = LogAnalyzer()
        result = analyzer.analyze(LogAnalysisRequest(
            log_content=open("test.log").read(),
            filename="test.log",
        ))
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        retriever: Retriever | None = None,
    ):
        """初始化 — 默认使用全局单例依赖，支持外部注入以减少资源占用"""
        if llm_client and retriever:
            self.llm_client = llm_client
            self.retriever = retriever
        else:
            from backend.api.deps import get_llm_client, get_rag_pipeline
            pipeline = get_rag_pipeline()
            self.llm_client = llm_client or get_llm_client()
            self.retriever = retriever or pipeline.retriever
        logger.info("日志分析 Agent 已初始化")

    def analyze(self, request: LogAnalysisRequest) -> AnalysisResult:
        """执行完整的日志分析流程

        Args:
            request: 日志分析请求（含日志内容和参数）

        Returns:
            结构化的故障分析报告
        """
        analysis_id = str(uuid.uuid4())[:8]
        log_content = request.log_content
        filename = request.filename

        logger.info(f"开始分析日志: {filename} (长度: {len(log_content)} 字符)")

        # ---- 步骤 1：异常信息提取 ----
        exceptions = self._extract_exceptions(log_content)
        logger.info(f"  检测到 {len(exceptions)} 个异常")

        # ---- 步骤 2：检索历史相似案例 ----
        historical_cases = []
        if request.include_historical and exceptions:
            historical_cases = self._search_historical_cases(
                exceptions, top_k=request.top_k
            )
            logger.info(f"  检索到 {len(historical_cases)} 个历史案例")

        # ---- 步骤 3：构建分析 Prompt 并调用 LLM ----
        # 格式化异常信息为文本
        exceptions_text = self._format_exceptions(exceptions)
        historical_text = self._format_historical_cases(historical_cases)

        template = get_template("log_analysis")
        messages = [
            {"role": "system", "content": template.system},
            {"role": "user", "content": template.user.format(
                log_content=log_content[:8000],  # 限制长度，避免超出 token 限制
                historical_cases=historical_text or "未找到相关历史案例",
            )},
        ]

        llm_response = self.llm_client.chat(
            messages=messages,
            temperature=template.temperature,
        )

        # ---- 步骤 4：解析 LLM 输出 ----
        result = self._parse_llm_analysis(
            llm_response=llm_response,
            analysis_id=analysis_id,
            filename=filename,
            exceptions=exceptions,
            historical_cases=historical_cases,
        )

        logger.info(f"日志分析完成: {analysis_id}")
        return result

    # ==================== 异常提取 ====================

    def _extract_exceptions(self, log_content: str) -> list[ExceptionInfo]:
        """从日志文本中提取异常信息

        支持识别 Traceback 格式和单行异常日志
        """
        exceptions = []

        # 模式1：标准 Python Traceback
        # Traceback (most recent call last):
        #   File "xxx.py", line N, in func
        #     code...
        # ExceptionType: message
        traceback_pattern = re.compile(
            # 捕获 Traceback 块，直到空行或字符串结尾；
            # 之前的 (?=\n(?=[A-Z]\w+:)) 会在 "AuthError: ..." 之前截断，导致漏掉异常行。
            r'Traceback\s*\(most recent call last\):(.*?)(?=\n\n|\Z)',
            re.DOTALL,
        )
        for match in traceback_pattern.finditer(log_content):
            tb_block = match.group(0).strip()
            exc = self._parse_traceback_block(tb_block)
            if exc:
                exceptions.append(exc)

        # 模式2：单独的异常行（非 Traceback 场景）
        # 格式: ExceptionName: error message
        for exc_type in KNOWN_EXCEPTIONS:
            pattern = re.compile(
                rf'({exc_type})[：:]\s*(.*?)(?=\n|$)',
                re.MULTILINE,
            )
            for match in pattern.finditer(log_content):
                exc = ExceptionInfo(
                    exception_type=match.group(1),
                    message=match.group(2).strip()[:500],
                )
                # 避免重复添加
                if not any(
                    e.exception_type == exc.exception_type
                    and e.message[:50] == exc.message[:50]
                    for e in exceptions
                ):
                    exceptions.append(exc)

        # 模式3：Selenium 常见异常日志
        selenium_patterns = [
            (r'TimeoutException', r'(?:Timeout|timed?\s*out)[：:]\s*(.*?)(?=\n|$)'),
            (r'NoSuchElementException', r'(?:Unable to locate element|no such element)[：:]\s*(.*?)(?=\n|$)'),
            (r'ConnectionError', r'(?:Connection refused|Failed to establish)[：:]\s*(.*?)(?=\n|$)'),
        ]
        for exc_type, pattern in selenium_patterns:
            for match in re.finditer(pattern, log_content, re.IGNORECASE):
                exc = ExceptionInfo(
                    exception_type=exc_type,
                    message=match.group(1).strip()[:500],
                )
                if not any(
                    e.exception_type == exc.exception_type
                    and e.message[:50] == exc.message[:50]
                    for e in exceptions
                ):
                    exceptions.append(exc)

        return exceptions

    def _parse_traceback_block(self, tb_text: str) -> ExceptionInfo | None:
        """解析单个 Traceback 文本块，提取异常类型、消息和文件行号"""
        # 提取最后的异常行
        exc_line_match = re.search(
            r'(\w+(?:Error|Exception|Warning))\s*[：:]\s*(.*)',
            tb_text,
        )
        if not exc_line_match:
            return None

        exc_type = exc_line_match.group(1)
        exc_message = exc_line_match.group(2).strip()[:500]

        # 提取文件路径和行号
        file_match = re.search(
            r'File\s+"([^"]+)",\s*line\s+(\d+)',
            tb_text,
        )
        file_path = ""
        line_number = None
        if file_match:
            file_path = file_match.group(1)
            line_number = int(file_match.group(2))

        # 提取关键行
        lines = [line for line in tb_text.split("\n") if line.strip()]
        key_lines = lines[-5:] if len(lines) > 5 else lines  # 最后5行最关键

        return ExceptionInfo(
            exception_type=exc_type,
            message=exc_message,
            traceback_lines=key_lines,
            file_path=file_path,
            line_number=line_number,
        )

    # ==================== 历史案例检索 ====================

    def _search_historical_cases(
        self,
        exceptions: list[ExceptionInfo],
        top_k: int = 3,
    ) -> list[HistoricalCase]:
        """在知识库中检索与当前异常相似的历史案例

        Args:
            exceptions: 当前检测到的异常列表
            top_k: 每个异常检索的案例数

        Returns:
            历史案例列表（按相似度降序，去重）
        """
        all_cases = []

        for exc in exceptions:
            # 组合异常类型和消息作为查询
            query = f"{exc.exception_type}: {exc.message[:200]}"

            try:
                docs = self.retriever.similarity_search(
                    query=query,
                    top_k=top_k,
                )
                for doc in docs:
                    score = doc.metadata.get("score", 0.0)
                    if score > 0.3:  # 只保留相似度 > 0.3 的结果
                        all_cases.append(HistoricalCase(
                            case_id=doc.metadata.get("chunk_id", ""),
                            title=f"{exc.exception_type} - {exc.message[:80]}",
                            similarity_score=round(score, 4),
                            description=doc.page_content[:500],
                            root_cause="",
                            solution="",
                            source=doc.metadata.get("filename", "knowledge_base"),
                        ))
            except Exception as e:
                logger.warning(f"检索历史案例失败 ({exc.exception_type}): {e}")

        # 按相似度降序排列，去重
        seen_contents = set()
        unique_cases = []
        for case in sorted(all_cases, key=lambda x: x.similarity_score, reverse=True):
            key = case.description[:100]
            if key not in seen_contents:
                seen_contents.add(key)
                unique_cases.append(case)

        return unique_cases[:top_k * 2]

    # ==================== 结果格式化 ====================

    def _format_exceptions(self, exceptions: list[ExceptionInfo]) -> str:
        """将异常列表格式化为文本"""
        if not exceptions:
            return "未检测到已知异常类型"

        parts = []
        for i, exc in enumerate(exceptions, 1):
            parts.append(
                f"异常 {i}: {exc.exception_type}\n"
                f"  消息: {exc.message}\n"
                f"  位置: {exc.file_path}:{exc.line_number or '?'}"
            )
        return "\n\n".join(parts)

    def _format_historical_cases(self, cases: list[HistoricalCase]) -> str:
        """将历史案例列表格式化为文本"""
        if not cases:
            return ""

        parts = []
        for i, case in enumerate(cases, 1):
            parts.append(
                f"案例 {i} (相似度: {case.similarity_score:.2f}):\n"
                f"  来源: {case.source}\n"
                f"  内容: {case.description[:300]}"
            )
        return "\n\n".join(parts)

    def _parse_llm_analysis(
        self,
        llm_response: str,
        analysis_id: str,
        filename: str,
        exceptions: list[ExceptionInfo],
        historical_cases: list[HistoricalCase],
    ) -> AnalysisResult:
        """解析 LLM 返回的分析文本，提取各部分内容"""

        # 尝试从 LLM 输出中解析各段落
        summary = self._extract_section(llm_response, "问题摘要", "")
        causes = self._extract_section_list(llm_response, "可能原因")
        fixes = self._extract_section_list(llm_response, "修复建议")

        # 如果解析失败，使用整个 LLM 输出作为摘要
        if not summary:
            # 取前 300 个字符作为摘要
            summary = llm_response[:300].strip()

        # 评估严重等级
        severity = self._assess_severity(exceptions)

        return AnalysisResult(
            analysis_id=analysis_id,
            filename=filename,
            summary=summary,
            exceptions_found=exceptions,
            possible_causes=causes,
            historical_cases=historical_cases,
            fix_suggestions=fixes,
            severity=severity,
            raw_analysis=llm_response,
        )

    @staticmethod
    def _extract_section(text: str, section_name: str, default: str = "") -> str:
        """从 markdown 格式文本中提取指定段落"""
        pattern = rf'###\s*{section_name}\s*\n(.*?)(?=\n###|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return default

    @staticmethod
    def _extract_section_list(text: str, section_name: str) -> list[str]:
        """从 markdown 格式文本中提取列表型段落"""
        content = LogAnalyzer._extract_section(text, section_name)
        if not content:
            return []

        # 提取每行列表项
        items = re.findall(r'[-*]\s*(.*)', content)
        return [item.strip() for item in items if item.strip()]

    @staticmethod
    def _assess_severity(exceptions: list[ExceptionInfo]) -> str:
        """根据异常类型和数量评估严重等级"""
        if not exceptions:
            return "低"

        high_severity = {"ConnectionError", "SQLException", "OperationalError"}
        medium_severity = {"TimeoutException", "NoSuchElementException", "AssertionError"}

        exc_types = {exc.exception_type for exc in exceptions}

        if exc_types & high_severity:
            return "高"
        elif exc_types & medium_severity or len(exceptions) > 3:
            return "中"
        return "低"
