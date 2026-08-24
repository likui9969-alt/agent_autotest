"""
Redis 缓存后端
==============

多进程共享缓存后端（docker-compose 部署时推荐）：
- 连接失败在工厂（backend/cache/__init__.py get_cache）中捕获并降级 MemoryCache，
  本类构造时 ping 校验连通性，失败直接抛异常。
- delete_prefix 用 SCAN 增量扫描（不阻塞 Redis，不用 KEYS *）。
"""
import logging

from backend.cache.base import CacheProtocol

logger = logging.getLogger("ai_rd_agent")


class RedisCache(CacheProtocol):
    """基于 redis-py 同步客户端的缓存后端"""

    def __init__(self, url: str, default_ttl: int = 300):
        import redis  # 延迟导入：未安装/未启用时不影响其他功能

        self._default_ttl = default_ttl
        self._client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        # 连通性校验：失败抛异常，由工厂降级
        self._client.ping()
        logger.info(f"Redis 缓存后端已连接: {url}")

    def get(self, key: str) -> str | None:
        try:
            return self._client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET 失败（视为未命中）: {e}")
            return None

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        try:
            self._client.set(key, value, ex=ttl if ttl is not None else self._default_ttl)
        except Exception as e:
            logger.warning(f"Redis SET 失败（忽略）: {e}")

    def delete_prefix(self, prefix: str) -> None:
        """按前缀删除 — SCAN 增量扫描，避免 KEYS 阻塞"""
        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=f"{prefix}*", count=100)
                if keys:
                    deleted += self._client.delete(*keys)
                if cursor == 0:
                    break
            if deleted:
                logger.info(f"Redis 缓存前缀失效: {prefix}* 删除 {deleted} 个键")
        except Exception as e:
            logger.warning(f"Redis 前缀删除失败（忽略）: {e}")
