"""
页面知识持久化
==============
存储网站页面探索结果，支持缓存复用、变化检测和语义检索。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.config.settings import get_settings
from backend.models.page import PageKnowledge
from backend.rag.vector_store import VectorStore

logger = logging.getLogger("ai_rd_agent")

# 页面知识 30 天过期（秒）
_PAGE_TTL_SECONDS = 30 * 24 * 3600


class PageKnowledgeStore:
    """页面知识存储

    同时使用 SQLite 保存结构化元数据，使用 Chroma 集合 page_knowledge
    保存向量化的页面摘要，支持按 URL 快速查找和语义检索。
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
        self._vector_store = VectorStore(collection_name="page_knowledge")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS page_knowledge (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT,
                    page_type TEXT,
                    page_hash TEXT,
                    content_json TEXT,
                    html_summary TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    access_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_page_knowledge_url
                ON page_knowledge(url)
            """)
            conn.commit()

    def _get_embedder(self):
        """获取嵌入模型（复用 RAG Pipeline 的 embedder）"""
        from backend.api.deps import get_rag_pipeline
        return get_rag_pipeline().embedder

    def get_by_url(self, url: str) -> Optional[PageKnowledge]:
        """按 URL 读取页面知识"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM page_knowledge WHERE url = ?",
                (url,),
            ).fetchone()
        return self._row_to_page(row) if row else None

    def is_expired(self, page: PageKnowledge) -> bool:
        """判断页面知识是否过期"""
        try:
            updated = datetime.fromisoformat(page.updated_at)
            return (datetime.now() - updated).total_seconds() > _PAGE_TTL_SECONDS
        except Exception:
            return True

    def save(self, page: PageKnowledge) -> str:
        """保存或更新页面知识"""
        page.id = page.page_hash or self._compute_hash(page.url)
        now = datetime.now().isoformat()
        if not page.created_at:
            page.created_at = now
        page.updated_at = now

        content_json = json.dumps(page.model_dump(), ensure_ascii=False, default=str)

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM page_knowledge WHERE url = ?",
                (page.url,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE page_knowledge
                    SET id = ?, title = ?, page_type = ?, page_hash = ?, content_json = ?,
                        html_summary = ?, updated_at = ?, access_count = access_count + 1
                    WHERE url = ?
                    """,
                    (
                        page.id, page.title, page.page_type, page.page_hash,
                        content_json, page.html_summary, now, page.url,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO page_knowledge
                    (id, url, title, page_type, page_hash, content_json, html_summary, created_at, updated_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        page.id, page.url, page.title, page.page_type, page.page_hash,
                        content_json, page.html_summary, now, now, page.access_count,
                    ),
                )
            conn.commit()

        # 写入向量库
        try:
            self._embed_and_store(page)
        except Exception as e:
            logger.warning(f"页面知识向量写入失败: {e}")

        return page.id

    def _embed_and_store(self, page: PageKnowledge):
        """将页面摘要嵌入并写入向量库"""
        from langchain_core.documents import Document

        summary = page.summary_text()
        embedder = self._get_embedder()
        embeddings = embedder.embed_documents([summary])
        doc = Document(
            page_content=summary,
            metadata={
                "url": page.url,
                "page_hash": page.page_hash,
                "updated_at": page.updated_at,
            },
        )
        # 先删除旧向量
        try:
            self._vector_store.delete_document(page.url)
        except Exception:
            pass
        self._vector_store.add_documents([doc], embeddings)

    def search(self, query: str, top_k: int = 3) -> list[PageKnowledge]:
        """语义搜索相关页面"""
        embedder = self._get_embedder()
        embedding = embedder.embed_query(query)
        results = self._vector_store.query(embedding, top_k=top_k)
        pages = []
        for meta in results.get("metadatas", [[]])[0]:
            url = meta.get("url")
            page = self.get_by_url(url)
            if page:
                pages.append(page)
        return pages

    def list_pages(self, limit: int = 100) -> list[PageKnowledge]:
        """列出所有页面知识"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM page_knowledge ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_page(row) for row in rows]

    def delete_page(self, url: str) -> bool:
        """删除页面知识"""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM page_knowledge WHERE url = ?",
                (url,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_page(self, row: sqlite3.Row) -> PageKnowledge:
        data = json.loads(row["content_json"])
        data["id"] = row["id"]
        data["url"] = row["url"]
        data["access_count"] = row["access_count"]
        return PageKnowledge(**data)

    @staticmethod
    def _compute_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()


# 全局单例
_page_store: Optional[PageKnowledgeStore] = None


def get_page_knowledge_store() -> PageKnowledgeStore:
    """获取页面知识存储单例"""
    global _page_store
    if _page_store is None:
        _page_store = PageKnowledgeStore()
    return _page_store


def compute_page_hash(page_info: dict) -> str:
    """根据页面信息计算稳定哈希"""
    content = json.dumps(page_info, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode("utf-8")).hexdigest()
