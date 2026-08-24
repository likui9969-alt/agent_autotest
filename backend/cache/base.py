"""
缓存后端协议与空实现
====================

所有缓存后端实现 CacheProtocol：
- get(key) → 缓存值（str）或 None
- set(key, value, ttl) → 写入（ttl 秒后过期）
- delete_prefix(prefix) → 删除所有以 prefix 开头的键（知识库变更时批量失效）

值统一为 str（调用方自行 JSON 序列化/反序列化），保持后端无关。
"""
import logging

logger = logging.getLogger("ai_rd_agent")


class CacheProtocol:
    """缓存后端接口（duck-typing 协议，无需显式继承）"""

    def get(self, key: str) -> str | None:
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        raise NotImplementedError

    def delete_prefix(self, prefix: str) -> None:
        raise NotImplementedError


class NullCache:
    """空缓存 — 缓存禁用时的占位实现（所有操作 no-op）"""

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        pass

    def delete_prefix(self, prefix: str) -> None:
        pass
