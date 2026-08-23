"""
LLM 熔断器模块
实现 CLOSED → OPEN → HALF_OPEN 三态熔断模式，防止连续失败耗尽 API 额度。

状态机:
  CLOSED（正常）-- 连续失败达阈值 --> OPEN（熔断）
  OPEN（熔断）  -- 超时 RECOVERY_TIMEOUT --> HALF_OPEN（探测）
  HALF_OPEN（探测）-- 请求成功 → CLOSED
  HALF_OPEN（探测）-- 请求失败 → OPEN

线程安全：所有状态读写均通过 threading.Lock 保护，支持多线程并发调用。
"""
import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger("ai_rd_agent")


class CircuitBreakerOpenError(Exception):
    """熔断器开启时抛出的异常，表示 LLM 调用被快速拒绝"""


class CircuitState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5       # 连续失败 N 次后熔断
    recovery_timeout: float = 30.0   # OPEN 状态持续秒数后进入 HALF_OPEN
    success_threshold: int = 2       # HALF_OPEN 下连续成功 N 次恢复


class CircuitBreaker:
    """熔断器（线程安全）

    LLMClient 作为 @lru_cache 全局单例被 FastAPI 多并发请求共享，
    因此熔断器的状态读写必须加锁，防止竞态导致状态错乱。
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_open(self) -> bool:
        """熔断器是否开启（直接拒绝请求）"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否应该进入 HALF_OPEN
                if time.time() - self._last_failure_time >= self._config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info("熔断器: OPEN → HALF_OPEN (恢复超时)")
                    return False
                return True
            return False

    def record_success(self):
        """记录一次成功调用"""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    logger.info("熔断器: HALF_OPEN → CLOSED (恢复成功)")

    def record_failure(self):
        """记录一次失败调用"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._success_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"熔断器: HALF_OPEN → OPEN (探测请求失败, "
                    f"将等待 {self._config.recovery_timeout}s)"
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._config.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(
                    f"熔断器: CLOSED → OPEN (连续 {self._failure_count} 次失败, "
                    f"将等待 {self._config.recovery_timeout}s 后尝试恢复)"
                )

    def reset(self):
        """手动重置为 CLOSED 状态"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0.0
            logger.info("熔断器: 手动重置为 CLOSED")
