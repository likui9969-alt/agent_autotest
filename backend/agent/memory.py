"""
对话记忆模块
为 Agent 提供多轮对话上下文记忆，支持环形缓冲区、过期清理和会话隔离
"""
import logging
import time
from collections import OrderedDict

logger = logging.getLogger("ai_rd_agent")

# 默认最大对话轮次
_DEFAULT_MAX_TURNS = 10

# Session 不活跃超时（1 小时）
_SESSION_TTL_SECONDS = 3600


class ConversationMemory:
    """对话记忆 — 环形缓冲区

    保存最近 N 轮对话历史，支持格式化为系统提示上下文。

    使用示例：
        memory = ConversationMemory(max_turns=10)
        memory.add_turn("你好", "你好！有什么可以帮助你的？")
        context = memory.format_context()
    """

    def __init__(self, max_turns: int = _DEFAULT_MAX_TURNS):
        """
        Args:
            max_turns: 最大保留的对话轮次（超过时自动覆盖最早的）
        """
        self._max_turns = max_turns
        self._history: list[dict] = []
        self._last_access = time.time()

    def add_turn(self, user_input: str, agent_response: str) -> None:
        """追加一轮对话

        Args:
            user_input: 用户本轮输入
            agent_response: Agent 本轮回复
        """
        self._history.append({
            "user": user_input,
            "assistant": agent_response,
            "timestamp": time.time(),
        })
        # 超过最大轮次时移除最早的
        while len(self._history) > self._max_turns:
            self._history.pop(0)
        self._last_access = time.time()

    def get_history(self, limit: int | None = None) -> list[dict]:
        """获取最近 N 轮对话历史

        Args:
            limit: 返回轮次数（默认全部）

        Returns:
            对话历史列表 [{user, assistant, timestamp}, ...]
        """
        self._last_access = time.time()
        if limit is not None:
            return self._history[-limit:]
        return list(self._history)

    def format_context(self, limit: int = 5) -> str:
        """将对话历史格式化为系统提示上下文字符串

        Args:
            limit: 最多包含的最近轮次

        Returns:
            格式化后的历史文本（空历史返回空字符串）
        """
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
    """Session 记忆管理器

    使用 LRU 缓存管理多个会话的对话记忆，自动清理过期会话。

    使用示例：
        manager = SessionMemoryManager()
        memory = manager.get_or_create("session-abc")
        memory.add_turn("你好", "你好！")
    """

    def __init__(self, max_sessions: int = 100):
        """
        Args:
            max_sessions: 最多同时管理的会话数
        """
        self._sessions: OrderedDict[str, ConversationMemory] = OrderedDict()
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> ConversationMemory:
        """获取或创建会话记忆

        Args:
            session_id: 会话唯一标识

        Returns:
            ConversationMemory 实例
        """
        now = time.time()

        # 清理过期会话（每获取一次清理一批）
        self._evict_expired()

        if session_id in self._sessions:
            # 移到末尾（LRU）
            memory = self._sessions.pop(session_id)
            self._sessions[session_id] = memory
            return memory

        # 超过最大会话数时淘汰最旧的
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
