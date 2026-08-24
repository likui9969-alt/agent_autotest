"""
缓存层测试
===========
- MemoryCache：set/get、TTL 过期、LRU 驱逐、前缀删除
- get_cache 工厂：禁用 → NullCache；redis 初始化失败 → 降级 MemoryCache
- RAG 路由缓存集成：命中不重查、不同参数不串缓存、知识库变更后失效
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from backend.cache import get_cache, reset_cache
from backend.cache.base import NullCache
from backend.cache.memory_cache import MemoryCache


class TestMemoryCache:
    """MemoryCache 单元测试"""

    def test_set_get(self):
        cache = MemoryCache(default_ttl=60)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_miss_returns_none(self):
        cache = MemoryCache()
        assert cache.get("nope") is None

    def test_ttl_expiry(self):
        cache = MemoryCache(default_ttl=1)
        cache.set("k1", "v1", ttl=0)  # 立即过期
        assert cache.get("k1") is None

    def test_ttl_lazy_expiry_with_monotonic(self):
        """写入后时间前进，过期条目 get 时返回 None"""
        cache = MemoryCache(default_ttl=60)
        cache.set("k1", "v1", ttl=10)
        # 模拟 11 秒后
        with patch("backend.cache.memory_cache.time.monotonic", return_value=time.monotonic() + 11):
            assert cache.get("k1") is None

    def test_lru_eviction(self):
        """容量满时淘汰最久未访问项"""
        cache = MemoryCache(max_size=2, default_ttl=60)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.get("k1")          # k1 变为最近访问
        cache.set("k3", "v3")    # 超容量，淘汰 k2（最久未访问）
        assert cache.get("k2") is None
        assert cache.get("k1") == "v1"
        assert cache.get("k3") == "v3"

    def test_delete_prefix(self):
        """按前缀删除不影响其他键"""
        cache = MemoryCache(default_ttl=60)
        cache.set("rag:query:a", "1")
        cache.set("rag:query:b", "2")
        cache.set("other:c", "3")

        cache.delete_prefix("rag:query:")

        assert cache.get("rag:query:a") is None
        assert cache.get("rag:query:b") is None
        assert cache.get("other:c") == "3"


class TestCacheFactory:
    """get_cache 工厂测试"""

    def setup_method(self):
        reset_cache()

    def teardown_method(self):
        reset_cache()

    def _settings(self, enabled=True, backend="memory"):
        s = MagicMock()
        s.CACHE_ENABLED = enabled
        s.CACHE_BACKEND = backend
        s.CACHE_TTL_SECONDS = 60
        s.CACHE_MAX_SIZE = 100
        s.REDIS_URL = "redis://localhost:6379/0"
        return s

    def test_disabled_returns_null_cache(self):
        with patch(
            "backend.config.settings.get_settings",
            return_value=self._settings(enabled=False),
        ):
            cache = get_cache()
            assert isinstance(cache, NullCache)
            cache.set("k", "v")
            assert cache.get("k") is None

    def test_memory_backend_default(self):
        with patch(
            "backend.config.settings.get_settings",
            return_value=self._settings(backend="memory"),
        ):
            cache = get_cache()
            assert isinstance(cache, MemoryCache)

    def test_redis_init_failure_falls_back_to_memory(self):
        """Redis 初始化失败应降级 MemoryCache（不抛异常）"""
        with patch(
            "backend.config.settings.get_settings",
            return_value=self._settings(backend="redis"),
        ), patch(
            "redis.Redis.from_url",
            side_effect=ConnectionError("redis unreachable"),
        ):
            cache = get_cache()
            assert isinstance(cache, MemoryCache)


class TestRagRouteCacheIntegration:
    """RAG 查询路由的缓存集成测试"""

    @pytest.fixture
    def client(self, monkeypatch):
        """轻量 TestClient — Fake pipeline + 内存缓存（每测试重置）"""
        reset_cache()
        from backend.main import app
        from backend.models.rag import RAGQueryResponse

        call_counter = {"count": 0}

        class FakePipeline:
            def query(self, request):
                call_counter["count"] += 1
                return RAGQueryResponse(
                    question=request.question,
                    answer=f"回答 #{call_counter['count']}",
                    sources=[],
                    retrieved_count=1,
                    rerank_used=False,
                    response_time_ms=1.0,
                )

            def rebuild(self, dir_path=None):
                return 5

        fake = FakePipeline()
        monkeypatch.setattr("backend.api.deps.get_rag_pipeline", lambda: fake)
        monkeypatch.setattr("backend.api.routes.rag.get_rag_pipeline", lambda: fake)

        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c, call_counter

        reset_cache()

    def test_second_identical_query_hits_cache(self, client):
        """相同问题第二次查询应命中缓存（pipeline 只调用一次）"""
        c, counter = client
        payload = {"question": "登录超时怎么办", "top_k": 3, "search_type": "similarity"}

        r1 = c.post("/api/v1/rag/query", json=payload)
        r2 = c.post("/api/v1/rag/query", json=payload)

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert counter["count"] == 1  # 第二次走缓存
        assert r1.json()["answer"] == r2.json()["answer"]

    def test_different_params_bypass_cache(self, client):
        """检索参数不同应绕过缓存"""
        c, counter = client
        base = {"question": "登录超时怎么办", "search_type": "similarity"}

        c.post("/api/v1/rag/query", json={**base, "top_k": 3})
        c.post("/api/v1/rag/query", json={**base, "top_k": 5})

        assert counter["count"] == 2

    def test_question_whitespace_normalized(self, client):
        """问题首尾空白归一化后仍命中缓存"""
        c, counter = client
        c.post("/api/v1/rag/query", json={"question": "登录超时怎么办", "search_type": "similarity"})
        c.post("/api/v1/rag/query", json={"question": "  登录超时怎么办  ", "search_type": "similarity"})

        assert counter["count"] == 1

    def test_rebuild_invalidates_cache(self, client):
        """知识库重建后缓存应失效（下次查询重新计算）"""
        c, counter = client
        payload = {"question": "登录超时怎么办", "search_type": "similarity"}

        c.post("/api/v1/rag/query", json=payload)
        assert counter["count"] == 1

        c.post("/api/v1/rag/rebuild")

        c.post("/api/v1/rag/query", json=payload)
        assert counter["count"] == 2  # 缓存已失效
