"""
持久化对话记忆存储
==================
基于 SQLite 保存多轮对话历史，支持会话隔离和过期清理。
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from backend.config.settings import get_settings

logger = logging.getLogger("ai_rd_agent")

# 默认最大保存轮次
_DEFAULT_MAX_TURNS = 20

# 默认无活动过期时间：30 分钟
_DEFAULT_SESSION_TTL_SECONDS = 30 * 60


class ConversationMemoryStore:
    """持久化对话记忆存储

    使用 SQLite 按 session_id 保存用户与 Agent 的对话轮次，
    保留最近 N 轮，超过 TTL 后视为过期。
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_turns: int = _DEFAULT_MAX_TURNS,
        ttl_seconds: int = _DEFAULT_SESSION_TTL_SECONDS,
    ):
        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds

        if db_path:
            self._db_path = Path(db_path)
        else:
            settings = get_settings()
            data_dir = Path(settings.get_log_dir()).parent
            data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = data_dir / "reports.db"

        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_input TEXT,
                    agent_response TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_session
                ON conversation_memory(session_id, created_at DESC)
            """)
            conn.commit()

    def add_turn(self, session_id: str, user_input: str, agent_response: str) -> None:
        """追加一轮对话到指定会话"""
        if not session_id:
            return

        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_memory (session_id, user_input, agent_response, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, user_input, agent_response, now),
            )
            conn.commit()

        self._trim_session(session_id)

    def _trim_session(self, session_id: str) -> None:
        """只保留最近 N 轮"""
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM conversation_memory
                WHERE session_id = ?
                  AND id NOT IN (
                      SELECT id FROM conversation_memory
                      WHERE session_id = ?
                      ORDER BY created_at DESC
                      LIMIT ?
                  )
                """,
                (session_id, session_id, self._max_turns),
            )
            conn.commit()

    def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        """获取指定会话最近 N 轮历史"""
        if not session_id:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_input, agent_response, created_at
                FROM conversation_memory
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [
            {
                "user": row["user_input"],
                "assistant": row["agent_response"],
                "timestamp": row["created_at"],
            }
            for row in reversed(rows)
        ]

    def format_context(self, session_id: str, limit: int = 5) -> str:
        """将会话历史格式化为系统提示上下文"""
        history = self.get_history(session_id, limit=limit)
        if not history:
            return ""

        parts = ["## 历史对话（最近几轮）"]
        for i, turn in enumerate(history, 1):
            user_msg = (turn["user"] or "")[:200]
            assistant_msg = (turn["assistant"] or "")[:500]
            parts.append(
                f"--- 第 {i} 轮 ---\n"
                f"用户: {user_msg}\n"
                f"助手: {assistant_msg}"
            )
        return "\n".join(parts)

    def is_expired(self, session_id: str) -> bool:
        """判断会话是否超过 TTL 未活动"""
        if not session_id:
            return True

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT created_at FROM conversation_memory
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        if not row:
            return True

        try:
            last_time = datetime.fromisoformat(row["created_at"])
            return (datetime.now() - last_time).total_seconds() > self._ttl_seconds
        except Exception:
            return True

    def clear(self, session_id: str) -> None:
        """清空指定会话的历史"""
        if not session_id:
            return

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_memory WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def count_turns(self, session_id: str) -> int:
        """统计会话轮次"""
        if not session_id:
            return 0

        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM conversation_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["cnt"] if row else 0


# 全局单例
_memory_store: Optional[ConversationMemoryStore] = None


def get_conversation_memory_store() -> ConversationMemoryStore:
    """获取持久化对话记忆存储单例"""
    global _memory_store
    if _memory_store is None:
        _memory_store = ConversationMemoryStore()
    return _memory_store
