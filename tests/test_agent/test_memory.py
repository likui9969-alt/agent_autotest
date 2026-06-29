"""
Tests for Conversation Memory Module
=====================================
- ConversationMemory: history management, format, expiration
- SessionMemoryManager: session isolation, eviction, auto-cleanup
"""
import time
from unittest.mock import patch, MagicMock
import pytest

from backend.agent.memory import (
    ConversationMemory,
    SessionMemoryManager,
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
