"""
测试报告持久化存储

使用 SQLite 保存测试执行结果，支持按时间查询和详情查看。
"""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.config.settings import get_settings


class TestReportStore:
    """测试报告存储

    使用 SQLite 保存测试执行结果，表结构：
        - id: 报告 UUID
        - name: 报告名称/场景名
        - report_type: 报告类型 (test/agent)
        - status: 状态 (passed/failed/error)
        - result_json: 完整结果 JSON
        - created_at: 创建时间
    """

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
                CREATE TABLE IF NOT EXISTS test_reports (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reports_created_at
                ON test_reports(created_at DESC)
            """)
            conn.commit()

    def save(
        self,
        name: str,
        report_type: str,
        status: str,
        result: dict,
        report_id: Optional[str] = None,
    ) -> str:
        """保存测试报告

        Args:
            name: 报告名称
            report_type: 报告类型 (test/agent)
            status: 状态 (passed/failed/error/timeout)
            result: 完整结果字典
            report_id: 可选指定 ID

        Returns:
            报告 ID
        """
        report_id = report_id or str(uuid.uuid4())[:8]
        created_at = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO test_reports (id, name, report_type, status, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    name,
                    report_type,
                    status,
                    json.dumps(result, ensure_ascii=False, default=str),
                    created_at,
                ),
            )
            conn.commit()
        return report_id

    def list_reports(self, limit: int = 50, report_type: Optional[str] = None) -> list[dict]:
        """查询报告列表"""
        with self._connect() as conn:
            if report_type:
                rows = conn.execute(
                    """
                    SELECT id, name, report_type, status, created_at
                    FROM test_reports
                    WHERE report_type = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (report_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, name, report_type, status, created_at
                    FROM test_reports
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [dict(row) for row in rows]

    def get_report(self, report_id: str) -> Optional[dict]:
        """获取报告详情"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM test_reports WHERE id = ?",
                (report_id,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["result"] = json.loads(result["result_json"])
        del result["result_json"]
        return result

    def delete_report(self, report_id: str) -> bool:
        """删除报告"""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM test_reports WHERE id = ?",
                (report_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
