"""
共享测试 fixtures 和 mock 工具
所有测试模块统一从此文件获取测试依赖。
"""
import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---- 确保 backend 包可导入 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==================== Mock 数据常量 ====================

SAMPLE_LOG_WITH_TRACEBACK = """
2024-01-15 10:30:45 | ERROR    | backend.selenium_driver:driver:342 | Exception in test
Traceback (most recent call last):
  File "backend/selenium_driver/scenarios/login.py", line 42, in run_login_test
    submit_btn = driver.find_element(By.ID, "login-btn")
  File "selenium/webdriver/remote/webdriver.py", line 856, in find_element
    return self.execute(Command.FIND_ELEMENT, {'using': by, 'value': value})['value']
  File "selenium/webdriver/remote/webdriver.py", line 439, in execute
    self.error_handler.check_response(response)
  File "selenium/webdriver/remote/errorhandler.py", line 242, in check_response
    raise exception_class(message, screen, stacktrace)
selenium.common.exceptions.TimeoutException: Message: timeout: Timed out waiting for element
  (Session info: headless chrome=x.x.x.x)
"""

SAMPLE_LOG_PLAIN = """
[2024-01-15 10:30:00] INFO: Starting login test
[2024-01-15 10:30:01] INFO: Opening page http://localhost:8000/demo
[2024-01-15 10:30:05] ERROR: TimeoutException: Timed out waiting for element
[2024-01-15 10:30:06] INFO: Test finished
"""

SAMPLE_LOG_NO_ERROR = """
[2024-01-15 10:30:00] INFO: Starting login test
[2024-01-15 10:30:01] INFO: Opening page http://localhost:8000/demo
[2024-01-15 10:30:02] INFO: Login successful
"""

SAMPLE_RUNTIME_LOG_LINES = [
    "2024-01-15 10:30:00 | INFO     | backend.main:42 | Server started\n",
    "2024-01-15 10:30:05 | ERROR    | backend.agent:99 | ConnectionError: Failed to connect\n",
    "2024-01-15 10:30:06 | WARNING  | backend.api:55 | Rate limit approaching\n",
    "2024-01-15 10:30:10 | INFO     | backend.main:50 | Health check OK\n",
]

SAMPLE_KNOWLEDGE_DOCS = [
    {
        "page_content": "登录超时通常由网络延迟或服务端响应慢引起。检查网络连接和服务健康状态。",
        "metadata": {"filename": "login_issues.txt", "score": 0.85, "chunk_id": "doc1"},
    },
    {
        "page_content": "数据库连接池耗尽会导致 OperationalError。需要增大连接池大小或优化查询。",
        "metadata": {"filename": "db_issues.txt", "score": 0.72, "chunk_id": "doc2"},
    },
]


# ==================== Settings Mock ====================

@pytest.fixture(autouse=True)
def mock_env_vars():
    """确保测试环境中关键的 env var 有默认值"""
    old = {}
    for k, v in [
        ("DASHSCOPE_API_KEY", "sk-test-key"),
        ("DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("LLM_MODEL", "deepseek-v3"),
        ("EMBEDDING_MODEL", "text-embedding-v3"),
    ]:
        old[k] = os.environ.get(k)
        os.environ[k] = v
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def mock_settings():
    """返回一个 Mock Settings 对象，替代真实 Pydantic 配置"""
    from unittest.mock import MagicMock

    settings = MagicMock()
    settings.APP_NAME = "TestApp"
    settings.APP_VERSION = "1.0.0"
    settings.DEBUG = True

    # LLM
    settings.DASHSCOPE_API_KEY = "sk-test-key"
    settings.DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    settings.LLM_MODEL = "deepseek-v3"
    settings.EMBEDDING_MODEL = "text-embedding-v3"
    settings.LLM_TEMPERATURE = 0.1
    settings.LLM_MAX_TOKENS = 4096

    # RAG
    settings.CHUNK_SIZE = 1000
    settings.CHUNK_OVERLAP = 200
    settings.RETRIEVER_TOP_K = 5
    settings.CHROMA_PERSIST_DIR = ""

    # Selenium
    settings.CHROMEDRIVER_PATH = ""
    settings.CHROME_BINARY_PATH = ""

    # JIRA
    settings.JIRA_URL = ""
    settings.JIRA_USERNAME = ""
    settings.JIRA_API_TOKEN = ""
    settings.JIRA_PROJECT_KEY = ""

    # Paths
    settings.DATA_DIR = ""
    settings.UPLOAD_DIR = ""
    settings.LOG_DIR = ""

    def fake_get_chroma_dir():
        return str(PROJECT_ROOT / "data" / "chroma")
    settings.get_chroma_dir = fake_get_chroma_dir

    def fake_get_log_dir():
        return str(PROJECT_ROOT / "data" / "logs")
    settings.get_log_dir = fake_get_log_dir

    def fake_get_upload_dir():
        return str(PROJECT_ROOT / "data" / "docs")
    settings.get_upload_dir = fake_get_upload_dir

    def fake_get_chromedriver_path():
        return ""
    settings.get_chromedriver_path = fake_get_chromedriver_path

    settings.Config = MagicMock()

    return settings


@pytest.fixture(autouse=True)
def patch_settings(mock_settings):
    """自动替换所有测试中的 get_settings() 为 mock 版本"""
    with patch("backend.config.settings.get_settings", return_value=mock_settings):
        yield


# ==================== LLM Client Mock ====================

@pytest.fixture
def mock_llm_client():
    """返回 Mock LLMClient"""
    client = MagicMock()

    def fake_chat(messages, temperature=None, max_tokens=None, stream=False):
        """模拟 chat 调用——根据输入返回固定回复"""
        return "这是一个 AI 生成的测试回复。"
    client.chat.side_effect = fake_chat

    def fake_embed(texts):
        """模拟 embed 调用——返回固定维度向量"""
        if not texts:
            return []
        return [[0.1] * 128 for _ in texts]
    client.embed.side_effect = fake_embed

    client.embed_single.return_value = [0.1] * 128

    def fake_chat_with_tools(messages, tools, tool_choice="auto", temperature=None, max_tokens=None):
        """模拟带工具的 chat 调用——根据消息内容判断是否返回 tool_call"""
        import json
        # 检查最后一条用户消息
        last_msg = messages[-1]["content"] if messages else ""
        # 如果 last_msg 含有工具调用指令的特定关键词，返回 tool_call
        if "SUPERVISOR: classify" in last_msg or "用户意图分类" in last_msg:
            return {"content": "", "tool_calls": None}
        return {"content": "AI analysis complete. No issues found.", "tool_calls": None}
    client.chat_with_tools.side_effect = fake_chat_with_tools

    return client


@pytest.fixture(autouse=True)
def patch_llm_client(mock_llm_client):
    """自动替换 deps.get_llm_client() 返回 mock LLMClient
    （不 mock LLMClient 类本身，这样 llm 测试可以测试真实类逻辑）
    """
    import backend.api.deps as deps_module
    with patch.object(deps_module, "get_llm_client", return_value=mock_llm_client):
        yield


# ==================== Agent/ReAct Mock ====================

@pytest.fixture
def mock_agent_state():
    """返回一个最小可用的 AgentState 字典"""
    from backend.agent.state import AgentState
    return {
        "messages": [],
        "task_type": "unknown",
        "user_input": "测试输入",
        "iteration_count": 0,
        "max_iterations": 5,
        "tool_calls": [],
        "tool_results": [],
        "context": {},
        "final_response": "",
        "next_action": "",
        "origin_node": "",
        "error": "",
    }


# ==================== VectorStore Mock ====================

class MockVectorStore:
    """替代 Chroma VectorStore 的内存实现"""

    def __init__(self):
        self._docs = {}
        self._collection_name = "test_kb"

    def add_documents(self, documents, embeddings):
        ids = []
        for doc in documents:
            doc_id = str(uuid.uuid4())
            self._docs[doc_id] = {"doc": doc, "embedding": [0.1] * 128}
            ids.append(doc_id)
        return ids

    def query(self, query_embedding, top_k=5, include_embeddings=False):
        items = list(self._docs.items())[:top_k]
        result = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        result["ids"][0] = [k for k, _ in items]
        result["documents"][0] = [v["doc"].page_content if hasattr(v["doc"], "page_content") else v["doc"] for v in (x[1] for x in items)]
        result["metadatas"][0] = [v["doc"].metadata if hasattr(v["doc"], "metadata") else {} for v in (x[1] for x in items)]
        result["distances"][0] = [0.15 * i for i in range(len(items))]
        if include_embeddings:
            result["embeddings"] = [[0.1] * 128 for _ in items]
        return result

    def get_stats(self):
        return {"collection_name": self._collection_name, "total_chunks": len(self._docs), "persist_directory": "/tmp/test"}

    def delete_collection(self):
        self._docs = {}

    def count(self):
        return len(self._docs)


@pytest.fixture
def mock_vector_store():
    """返回内存 MockVectorStore 实例"""
    return MockVectorStore()


# ==================== Retriever Mock ====================

@pytest.fixture
def mock_retriever():
    """返回 Mock Retriever"""
    from unittest.mock import MagicMock
    from langchain_core.documents import Document

    retriever = MagicMock()

    def fake_similarity_search(query, top_k=None):
        k = top_k or 5
        return [
            Document(
                page_content=doc["page_content"],
                metadata={**doc["metadata"], "score": doc["metadata"]["score"]},
            )
            for doc in SAMPLE_KNOWLEDGE_DOCS[:k]
        ]
    retriever.similarity_search.side_effect = fake_similarity_search

    def fake_mmr_search(query, top_k=None, lambda_mult=0.5):
        return fake_similarity_search(query, top_k)
    retriever.mmr_search.side_effect = fake_mmr_search

    return retriever


# ==================== RAG Pipeline Mock ====================

@pytest.fixture
def mock_rag_pipeline():
    """返回 Mock RAGPipeline"""
    from unittest.mock import MagicMock

    pipeline = MagicMock()
    pipeline.ingest_file.return_value = 5
    pipeline.ingest_directory.return_value = 20

    from backend.models.rag import RAGQueryResponse
    pipeline.query.return_value = RAGQueryResponse(
        question="测试问题",
        answer="RAG 检索后生成的回答。",
        sources=[],
        retrieved_count=3,
        response_time_ms=100.0,
    )

    pipeline.rebuild.return_value = 20
    pipeline.stats.return_value = {"total_chunks": 20}

    return pipeline


# ==================== Log Analyzer Mock ====================

@pytest.fixture
def mock_log_analyzer():
    """返回 Mock LogAnalyzer"""
    from unittest.mock import MagicMock
    from backend.models.analysis import AnalysisResult, ExceptionInfo

    analyzer = MagicMock()

    def fake_analyze(request):
        return AnalysisResult(
            analysis_id="test-001",
            filename=request.filename,
            summary="模拟: 检测到登录超时异常。",
            exceptions_found=[
                ExceptionInfo(exception_type="TimeoutException", message="Timed out waiting for element")
            ],
            possible_causes=["网络延迟", "服务响应慢"],
            fix_suggestions=["增加超时时间", "检查服务健康状态"],
            severity="中",
            raw_analysis="模拟分析结果",
        )
    analyzer.analyze.side_effect = fake_analyze

    return analyzer


# ==================== File System Mock ====================

@pytest.fixture
def temp_log_dir(tmp_path):
    """创建临时日志目录"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "app.log"
    log_file.write_text("".join(SAMPLE_RUNTIME_LOG_LINES), encoding="utf-8")
    return log_dir


@pytest.fixture
def temp_doc_dir(tmp_path):
    """创建临时文档目录"""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    txt_file = doc_dir / "test_doc.txt"
    txt_file.write_text("这是一份测试文档内容。", encoding="utf-8")
    return doc_dir


# ==================== 通用辅助函数 ====================

def make_supervisor_state(user_input: str, task_type: str = "unknown") -> dict:
    """快速构造一个干净的 Supervisor 输入状态"""
    return {
        "messages": [],
        "task_type": task_type,
        "user_input": user_input,
        "iteration_count": 0,
        "max_iterations": 5,
        "tool_calls": [],
        "tool_results": [],
        "context": {},
        "final_response": "",
        "next_action": "",
        "origin_node": "",
        "error": "",
        "session_id": "",
        "memory_context": "",
    }


def make_react_state(
    user_input: str = "测试",
    node_name: str = "rag_node",
    iteration: int = 0,
    messages: list | None = None,
    tool_calls: list | None = None,
    tool_results: list | None = None,
) -> dict:
    """快速构造 ReAct 节点输入状态"""
    return {
        "messages": messages or [],
        "task_type": "rag_query",
        "user_input": user_input,
        "iteration_count": iteration,
        "max_iterations": 5,
        "tool_calls": tool_calls or [],
        "tool_results": tool_results or [],
        "context": {},
        "final_response": "",
        "next_action": "llm_reason",
        "origin_node": node_name,
        "error": "",
        "session_id": "",
        "memory_context": "",
    }
