"""
内存缓存后端（TTL + 容量上限 LRU）
==================================

无外部依赖的默认缓存后端：
- TTL：惰性过期（get 时检查），写入时清扫
- 容量：超出 max_size 淘汰最久未访问项（OrderedDict LRU）

适用：单进程部署 / CI 测试 / Redis 不可用时的降级。
"""
import threading
import time
from collections import OrderedDict

from backend.cache.base import CacheProtocol


class MemoryCache(CacheProtocol):
    """线程安全的 TTL 内存缓存"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, expire_at = item
            if expire_at <= time.monotonic():
                # 惰性过期
                del self._store[key]
                return None
            # 访问即移到 LRU 尾部
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        expire_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            self._store[key] = (value, expire_at)
            self._store.move_to_end(key)
            # 超容量：先淘汰最久未访问项
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            expired = [k for k in self._store if k.startswith(prefix)]
            for k in expired:
                del self._store[k]
