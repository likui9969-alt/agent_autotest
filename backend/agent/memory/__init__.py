"""
对话记忆模块
============
为 Agent 提供多轮对话上下文记忆，支持内存缓冲和持久化存储。
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict

from backend.agent.memory.store import ConversationMemoryStore, get_conversation_memory_store
from backend.agent.memory.context import build_memory_context

logger = logging.getLogger("ai_rd_agent")

# 默认最大对话轮次
_DEFAULT_MAX_TURNS = 10

# Session 不活跃超时（1 小时）
_SESSION_TTL_SECONDS = 3600


class ConversationMemory:
    """对话记忆 — 环形缓冲区（内存版）

    保存最近 N 轮对话历史，支持格式化为系统提示上下文。

    使用示例：
        memory = ConversationMemory(max_turns=10)
        memory.add_turn("你好", "你好！有什么可以帮助你的？")
        context = memory.format_context()
    """

    def __init__(self, max_turns: int = _DEFAULT_MAX_TURNS):
        self._max_turns = max_turns
        self._history: list[dict] = []
        self._last_access = time.time()

    def add_turn(self, user_input: str, agent_response: str) -> None:
        """追加一轮对话"""
        self._history.append({
            "user": user_input,
            "assistant": agent_response,
            "timestamp": time.time(),
        })
        while len(self._history) > self._max_turns:
            self._history.pop(0)
        self._last_access = time.time()

    def get_history(self, limit: int | None = None) -> list[dict]:
        """获取最近 N 轮对话历史"""
        self._last_access = time.time()
        if limit is not None:
            return self._history[-limit:]
        return list(self._history)

    def format_context(self, limit: int = 5) -> str:
        """将对话历史格式化为系统提示上下文字符串"""
        if not self._history:
            return ""

        recent = self._history[-limit:]
        parts = ["## 历史对话（最近几轮）"]
        for i, turn in enumerate(recent, 1):
            user_msg = turn["user"][:200]
            assistant_msg = turn["assistant"][:500]
            parts.append(
                f"--- 第 {i} 轮 ---\n"
                f"用户: {user_msg}\n"
                f"助手: {assistant_msg}"
            )
        return "\n".join(parts)

    @property
    def is_expired(self) -> bool:
        """检查会话是否超过不活跃超时时间"""
        return (time.time() - self._last_access) > _SESSION_TTL_SECONDS

    def clear(self) -> None:
        """清空对话历史"""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        """当前历史轮次数"""
        return len(self._history)


class SessionMemoryManager:
    """Session 记忆管理器（内存 LRU 版）

    使用 LRU 缓存管理多个会话的对话记忆，自动清理过期会话。
    """

    def __init__(self, max_sessions: int = 100):
        self._sessions: OrderedDict[str, ConversationMemory] = OrderedDict()
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> ConversationMemory:
        """获取或创建会话记忆"""
        self._evict_expired()

        if session_id in self._sessions:
            memory = self._sessions.pop(session_id)
            self._sessions[session_id] = memory
            return memory

        if len(self._sessions) >= self._max_sessions:
            self._sessions.popitem(last=False)

        memory = ConversationMemory()
        self._sessions[session_id] = memory
        return memory

    def _evict_expired(self) -> None:
        """清理超过不活跃超时时间的会话"""
        expired_ids = [
            sid for sid, mem in self._sessions.items()
            if mem.is_expired
        ]
        for sid in expired_ids:
            self._sessions.pop(sid, None)
            logger.debug(f"清理过期会话: {sid}")

    @property
    def active_session_count(self) -> int:
        """当前活跃会话数"""
        return len(self._sessions)


__all__ = [
    "ConversationMemory",
    "SessionMemoryManager",
    "ConversationMemoryStore",
    "get_conversation_memory_store",
    "build_memory_context",
]
