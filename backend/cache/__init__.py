"""
缓存层统一入口
==============

get_cache() 返回进程级单例：
- CACHE_ENABLED=False          → NullCache（禁用）
- CACHE_BACKEND="memory"（默认）→ MemoryCache（TTL + LRU，无外部依赖）
- CACHE_BACKEND="redis"        → RedisCache；连接失败自动降级 MemoryCache 并告警

reset_cache() 仅供测试重置单例。
"""
import logging

from backend.cache.base import CacheProtocol, NullCache
from backend.cache.memory_cache import MemoryCache

logger = logging.getLogger("ai_rd_agent")

_cache_instance: CacheProtocol | None = None


def get_cache() -> CacheProtocol:
    """获取缓存后端单例（按配置初始化）"""
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    # 模块属性访问 get_settings，保证测试 patch 一致生效（同 llm/client.py 模式）
    from backend.config import settings as _settings_module

    settings = _settings_module.get_settings()

    if not settings.CACHE_ENABLED:
        logger.info("缓存已禁用（CACHE_ENABLED=False）")
        _cache_instance = NullCache()
        return _cache_instance

    if settings.CACHE_BACKEND == "redis":
        try:
            from backend.cache.redis_cache import RedisCache

            _cache_instance = RedisCache(
                url=settings.REDIS_URL,
                default_ttl=settings.CACHE_TTL_SECONDS,
            )
            return _cache_instance
        except Exception as e:
            logger.warning(
                f"Redis 缓存初始化失败，降级为内存缓存: {e}。"
                f"检查 REDIS_URL={settings.REDIS_URL} 或 docker compose 启动 redis 服务"
            )
            # 降级继续走 MemoryCache

    _cache_instance = MemoryCache(
        max_size=settings.CACHE_MAX_SIZE,
        default_ttl=settings.CACHE_TTL_SECONDS,
    )
    logger.info(
        f"内存缓存后端已启用（容量 {settings.CACHE_MAX_SIZE}，"
        f"TTL {settings.CACHE_TTL_SECONDS}s）"
    )
    return _cache_instance


def reset_cache() -> None:
    """重置缓存单例（仅测试使用）"""
    global _cache_instance
    _cache_instance = None
