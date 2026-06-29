"""
测试用例持久化存储
==================
使用 SQLite 保存自动生成的测试用例，支持增删改查和导出。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.config.settings import get_settings
from backend.models.test_case import TestCase


class TestCaseStore:
    """测试用例数据库存储"""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        if db_path:
            self._db_path = Path(db_path)
        else:
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
                CREATE TABLE IF NOT EXISTS test_cases (
                    id TEXT PRIMARY KEY,
                    module TEXT,
                    title TEXT NOT NULL,
                    objective TEXT,
                    preconditions TEXT,
                    steps_json TEXT,
                    expected_result TEXT,
                    priority TEXT,
                    source TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_cases_created_at
                ON test_cases(created_at DESC)
            """)
            conn.commit()

    def save(self, case: TestCase) -> str:
        """保存单个用例"""
        case_id = case.id or str(uuid.uuid4())[:8]
        case.id = case_id
        if not case.created_at:
            case.created_at = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO test_cases
                (id, module, title, objective, preconditions, steps_json, expected_result, priority, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case.id,
                    case.module,
                    case.title,
                    case.objective,
                    case.preconditions,
                    json.dumps(case.steps, ensure_ascii=False),
                    case.expected_result,
                    case.priority,
                    case.source,
                    case.created_at,
                ),
            )
            conn.commit()
        return case.id

    def save_many(self, cases: list[TestCase]) -> int:
        """批量保存"""
        count = 0
        for case in cases:
            self.save(case)
            count += 1
        return count

    def list_cases(self, limit: int = 100, offset: int = 0) -> list[TestCase]:
        """分页查询用例列表"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM test_cases
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_case(row) for row in rows]

    def get_case(self, case_id: str) -> Optional[TestCase]:
        """获取单条用例"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM test_cases WHERE id = ?",
                (case_id,),
            ).fetchone()
        return self._row_to_case(row) if row else None

    def delete_case(self, case_id: str) -> bool:
        """删除用例"""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM test_cases WHERE id = ?",
                (case_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        """统计用例总数"""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM test_cases").fetchone()
            return row["cnt"] if row else 0

    def _row_to_case(self, row: sqlite3.Row) -> TestCase:
        steps = json.loads(row["steps_json"]) if row["steps_json"] else []
        return TestCase(
            id=row["id"],
            module=row["module"],
            title=row["title"],
            objective=row["objective"],
            preconditions=row["preconditions"],
            steps=steps,
            expected_result=row["expected_result"],
            priority=row["priority"],
            source=row["source"],
            created_at=row["created_at"],
        )
