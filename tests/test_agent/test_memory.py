"""
Tests for Conversation Memory Module
=====================================
- ConversationMemory: history management, format, expiration
- SessionMemoryManager: session isolation, eviction, auto-cleanup
"""
import time
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from backend.agent.memory import (
    ConversationMemory,
    SessionMemoryManager,
    ConversationMemoryStore,
    _DEFAULT_MAX_TURNS,
    _SESSION_TTL_SECONDS,
)


class TestConversationMemory:
    """对话记忆测试"""

    def test_memory_basic(self):
        """添加两轮对话后应能获取历史"""
        memory = ConversationMemory(max_turns=10)
        memory.add_turn("你好", "你好！有什么可以帮助你的？")
        memory.add_turn("查一下登录超时", "正在为您查询知识库...")

        history = memory.get_history()
        assert len(history) == 2
        assert history[0]["user"] == "你好"
        assert history[1]["assistant"] == "正在为您查询知识库..."

    def test_memory_max_turns(self):
        """超过最大轮次时自动覆盖最早的"""
        memory = ConversationMemory(max_turns=3)
        for i in range(5):
            memory.add_turn(f"输入{i}", f"回复{i}")

        history = memory.get_history()
        assert len(history) == 3
        assert history[0]["user"] == "输入2"
        assert history[-1]["user"] == "输入4"

    def test_memory_format_context(self):
        """格式化输出应包含角色标记"""
        memory = ConversationMemory(max_turns=10)
        memory.add_turn("在吗", "在的")
        memory.add_turn("帮我查个bug", "好的")

        context = memory.format_context(limit=5)
        assert "历史对话" in context
        assert "用户: 在吗" in context
        assert "助手: 在的" in context
        assert "用户: 帮我查个bug" in context

    def test_memory_clear(self):
        """清空后应为空"""
        memory = ConversationMemory(max_turns=10)
        memory.add_turn("测试", "回复")
        memory.clear()

        assert memory.turn_count == 0
        assert memory.format_context() == ""

    def test_memory_empty_history(self):
        """无历史时 format_context 返回空字符串"""
        memory = ConversationMemory(max_turns=10)
        assert memory.format_context() == ""

    def test_memory_get_history_limit(self):
        """get_history 的 limit 参数应限制返回数量"""
        memory = ConversationMemory(max_turns=10)
        for i in range(5):
            memory.add_turn(f"输入{i}", f"回复{i}")

        limited = memory.get_history(limit=2)
        assert len(limited) == 2
        assert limited[0]["user"] == "输入3"

    def test_memory_expired(self):
        """超过不活跃时间应标记为过期"""
        memory = ConversationMemory(max_turns=10)
        memory.add_turn("test", "ok")

        # Mock 时间前进到过期后
        with patch.object(memory, "_last_access", time.time() - _SESSION_TTL_SECONDS - 1):
            assert memory.is_expired is True

    def test_memory_not_expired(self):
        """未超时应标记为不过期"""
        memory = ConversationMemory(max_turns=10)
        memory.add_turn("test", "ok")
        assert memory.is_expired is False


class TestSessionMemoryManager:
    """会话记忆管理器测试"""

    def test_get_or_create_new(self):
        """新的 session_id 应创建新记忆"""
        manager = SessionMemoryManager(max_sessions=10)
        memory = manager.get_or_create("session-abc")

        assert isinstance(memory, ConversationMemory)
        assert memory.turn_count == 0

    def test_get_or_create_reuse(self):
        """相同 session_id 应返回同一实例"""
        manager = SessionMemoryManager(max_sessions=10)
        mem1 = manager.get_or_create("session-abc")
        mem1.add_turn("你好", "你好")

        mem2 = manager.get_or_create("session-abc")
        assert mem2.turn_count == 1

    def test_session_isolation(self):
        """不同 session_id 的记忆应互不干扰"""
        manager = SessionMemoryManager(max_sessions=10)
        mem_a = manager.get_or_create("session-a")
        mem_b = manager.get_or_create("session-b")

        mem_a.add_turn("A的问题", "A的回复")
        mem_b.add_turn("B的问题", "B的回复")

        assert mem_a.turn_count == 1
        assert mem_b.turn_count == 1
        assert mem_a.get_history()[0]["user"] == "A的问题"
        assert mem_b.get_history()[0]["user"] == "B的问题"

    def test_max_sessions_eviction(self):
        """超过 max_sessions 时应淘汰最旧的会话"""
        manager = SessionMemoryManager(max_sessions=2)
        mem1 = manager.get_or_create("sess-1")
        mem2 = manager.get_or_create("sess-2")
        mem3 = manager.get_or_create("sess-3")

        # sess-1 应该被淘汰了
        mem1_again = manager.get_or_create("sess-1")
        # 因为淘汰了最旧的，sess-1 被永久移除了...
        # 实际行为：淘汰的是 LRU 最旧的（sess-2 不会移除非 LRU）
        # 当添加 sess-3 时，sess-1 是最旧的（最先访问），被淘汰
        # 所以 sess-1 会变成新创建，turn_count=0
        assert mem1_again is not mem1
        assert mem1_again.turn_count == 0

    def test_session_count(self):
        """active_session_count 应返回正确的会话数"""
        manager = SessionMemoryManager(max_sessions=10)
        assert manager.active_session_count == 0

        manager.get_or_create("sess-1")
        assert manager.active_session_count == 1

        manager.get_or_create("sess-2")
        assert manager.active_session_count == 2


class TestConversationMemoryStore:
    """持久化对话记忆存储测试"""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """使用临时数据库的记忆存储"""
        db_path = tmp_path / "memory_test.db"
        store = ConversationMemoryStore(db_path=str(db_path), max_turns=5)
        return store

    def test_add_turn_and_get_history(self, temp_store):
        """添加对话后能正确读取历史"""
        sid = str(uuid.uuid4())[:8]
        temp_store.add_turn(sid, "你好", "你好！")
        temp_store.add_turn(sid, "查日志", "正在查询...")

        history = temp_store.get_history(sid)
        assert len(history) == 2
        assert history[0]["user"] == "你好"
        assert history[1]["assistant"] == "正在查询..."

    def test_session_isolation_persistent(self, temp_store):
        """不同 session_id 在数据库中互相隔离"""
        sid_a = "sess-a"
        sid_b = "sess-b"
        temp_store.add_turn(sid_a, "A", "reply-A")
        temp_store.add_turn(sid_b, "B", "reply-B")

        assert len(temp_store.get_history(sid_a)) == 1
        assert len(temp_store.get_history(sid_b)) == 1
        assert temp_store.get_history(sid_a)[0]["user"] == "A"
        assert temp_store.get_history(sid_b)[0]["user"] == "B"

    def test_trim_max_turns(self, temp_store):
        """超过最大轮次后只保留最近 N 轮"""
        sid = "sess-trim"
        for i in range(10):
            temp_store.add_turn(sid, f"输入{i}", f"回复{i}")

        history = temp_store.get_history(sid, limit=10)
        assert len(history) == 5
        assert history[0]["user"] == "输入5"
        assert history[-1]["user"] == "输入9"

    def test_format_context(self, temp_store):
        """format_context 应返回带角色标记的字符串"""
        sid = "sess-ctx"
        temp_store.add_turn(sid, "在吗", "在的")
        context = temp_store.format_context(sid, limit=5)
        assert "历史对话" in context
        assert "用户: 在吗" in context
        assert "助手: 在的" in context

    def test_format_context_empty(self, temp_store):
        """无历史时 format_context 返回空字符串"""
        assert temp_store.format_context("no-such", limit=5) == ""

    def test_clear(self, temp_store):
        """清空后会话历史应为空"""
        sid = "sess-clear"
        temp_store.add_turn(sid, "test", "ok")
        assert temp_store.count_turns(sid) == 1

        temp_store.clear(sid)
        assert temp_store.count_turns(sid) == 0
        assert temp_store.get_history(sid) == []

    def test_is_expired(self, temp_store):
        """长时间未活动应判定为过期"""
        sid = "sess-expired"
        temp_store.add_turn(sid, "test", "ok")
        # 手动修改最近一轮时间为很久以前
        db_path = Path(temp_store._db_path)
        conn = __import__("sqlite3").connect(str(db_path))
        past = "2020-01-01T00:00:00"
        conn.execute(
            "UPDATE conversation_memory SET created_at = ? WHERE session_id = ?",
            (past, sid),
        )
        conn.commit()
        conn.close()

        assert temp_store.is_expired(sid) is True

    def test_not_expired(self, temp_store):
        """近期活动不应判定为过期"""
        sid = "sess-active"
        temp_store.add_turn(sid, "test", "ok")
        assert temp_store.is_expired(sid) is False
