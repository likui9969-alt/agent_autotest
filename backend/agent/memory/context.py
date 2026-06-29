"""
对话记忆上下文构建
==================
基于 ConversationMemoryStore 为 Agent 提供格式化的历史对话上下文。
"""
from __future__ import annotations

from backend.agent.memory.store import get_conversation_memory_store


def build_memory_context(session_id: str, limit: int = 5) -> str:
    """根据 session_id 从持久化存储中构建记忆上下文。

    Args:
        session_id: 会话唯一标识
        limit: 最多取最近几轮

    Returns:
        格式化后的历史对话字符串，无历史或 session_id 为空时返回空字符串
    """
    if not session_id:
        return ""

    store = get_conversation_memory_store()
    return store.format_context(session_id, limit=limit)
