"""
PageKnowledge 单元测试
======================
验证页面知识模型 summary_text 与 PageKnowledgeStore 的持久化/检索行为。
使用 mock 替换向量库与嵌入模型，不依赖真实 Chroma 服务。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from backend.models.page import PageElement, PageKnowledge
from backend.rag.page_knowledge import PageKnowledgeStore, _PAGE_TTL_SECONDS


class TestPageKnowledgeModel:
    def test_summary_text_empty_page(self):
        """空页面应生成基础摘要，不报错"""
        page = PageKnowledge(url="http://example.com")
        text = page.summary_text()
        assert "URL: http://example.com" in text
        assert "输入框: 0 个" in text
        assert "按钮: 0 个" in text

    def test_summary_text_with_elements(self):
        """summary_text 应正确访问 Pydantic 对象属性（而非 dict.get）"""
        page = PageKnowledge(
            url="http://example.com/login",
            title="登录页",
            page_type="login",
            inputs=[
                PageElement(name="username", type="text", placeholder="用户名"),
                PageElement(name="password", type="password"),
            ],
            buttons=[
                PageElement(text="登录", type="submit"),
                PageElement(text="注册", type="button"),
            ],
        )
        text = page.summary_text()
        assert "登录页" in text
        assert "页面类型: login" in text
        assert "username(text)" in text
        assert "password(password)" in text
        assert "登录" in text
        assert "注册" in text

    def test_summary_text_fallback_for_empty_name(self):
        """输入框 name/id/placeholder 都为空时不应崩溃"""
        page = PageKnowledge(
            url="http://example.com/form",
            inputs=[PageElement(type="text")],
        )
        text = page.summary_text()
        assert "输入框详情: (text)" in text

    def test_summary_text_includes_html_summary(self):
        """html_summary 应被截断后包含在摘要中"""
        long_summary = "x" * 600
        page = PageKnowledge(
            url="http://example.com",
            html_summary=long_summary,
        )
        text = page.summary_text()
        assert "摘要: " in text
        assert len(text.split("摘要: ")[1]) == 500


@pytest.fixture
def page_store(tmp_path):
    """返回使用临时 SQLite 且向量写入被 mock 的 PageKnowledgeStore"""
    db_path = tmp_path / "page_knowledge_test.db"

    with patch("backend.rag.page_knowledge.VectorStore") as mock_vs_cls:
        mock_vs_cls.return_value = MagicMock()

        store = PageKnowledgeStore(db_path=str(db_path))
        # 屏蔽向量嵌入，避免依赖真实 embedder
        store._embed_and_store = MagicMock()
        yield store


class TestPageKnowledgeStore:
    def test_save_and_get_by_url(self, page_store):
        """保存后应能按 URL 读取"""
        page = PageKnowledge(
            url="http://example.com/login",
            title="登录页",
            page_type="login",
        )
        page_id = page_store.save(page)

        loaded = page_store.get_by_url("http://example.com/login")
        assert loaded is not None
        assert loaded.id == page_id
        assert loaded.title == "登录页"
        assert loaded.page_type == "login"

    def test_save_updates_existing_page_and_increments_access_count(self, page_store):
        """重复保存同一 URL 应更新并增加 access_count"""
        url = "http://example.com/search"
        page = PageKnowledge(url=url, title="搜索页", page_type="search")

        page_store.save(page)
        first = page_store.get_by_url(url)
        assert first.access_count == 1

        page_store.save(page)
        second = page_store.get_by_url(url)
        assert second.access_count == 2
        assert second.title == "搜索页"

    def test_get_by_url_returns_none_for_missing(self, page_store):
        """未保存的 URL 应返回 None"""
        assert page_store.get_by_url("http://not.exist") is None

    def test_list_pages_ordered_by_updated(self, page_store):
        """list_pages 应按 updated_at 倒序返回"""
        page_store.save(PageKnowledge(url="http://a.com", title="A"))
        page_store.save(PageKnowledge(url="http://b.com", title="B"))

        pages = page_store.list_pages(limit=10)
        assert len(pages) == 2
        assert pages[0].title == "B"
        assert pages[1].title == "A"

    def test_list_pages_respects_limit(self, page_store):
        """list_pages 应遵守 limit 参数"""
        for i in range(5):
            page_store.save(PageKnowledge(url=f"http://site{i}.com"))
        assert len(page_store.list_pages(limit=2)) == 2

    def test_delete_page(self, page_store):
        """delete_page 应删除指定 URL 并返回影响行数"""
        url = "http://example.com/temp"
        page_store.save(PageKnowledge(url=url))
        assert page_store.get_by_url(url) is not None

        assert page_store.delete_page(url) is True
        assert page_store.get_by_url(url) is None
        assert page_store.delete_page(url) is False

    def test_is_expired_for_old_page(self, page_store):
        """updated_at 超过 TTL 的页面应判定为过期"""
        old_time = (datetime.now() - timedelta(seconds=_PAGE_TTL_SECONDS + 1)).isoformat()
        page = PageKnowledge(
            url="http://example.com/old",
            updated_at=old_time,
        )
        assert page_store.is_expired(page) is True

    def test_is_expired_for_recent_page(self, page_store):
        """updated_at 在 TTL 内的页面不应过期"""
        recent_time = datetime.now().isoformat()
        page = PageKnowledge(
            url="http://example.com/recent",
            updated_at=recent_time,
        )
        assert page_store.is_expired(page) is False

    def test_is_expired_for_invalid_timestamp(self, page_store):
        """异常时间戳应被视作过期"""
        page = PageKnowledge(
            url="http://example.com/bad",
            updated_at="not-a-timestamp",
        )
        assert page_store.is_expired(page) is True
