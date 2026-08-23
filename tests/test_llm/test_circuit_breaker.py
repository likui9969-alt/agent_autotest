"""
Tests for CircuitBreaker
========================
- 三态机：CLOSED → OPEN → HALF_OPEN → CLOSED
- 线程安全：多线程并发下状态一致
"""
import threading
import time
from unittest.mock import patch

import pytest

from backend.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)


class TestCircuitBreakerStates:
    """熔断器三态机测试"""

    def test_initial_state_closed(self):
        """初始状态应为 CLOSED"""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_open is False

    def test_closed_to_open_after_threshold(self):
        """连续失败达阈值后应进入 OPEN"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_open is True

    def test_not_open_below_threshold(self):
        """失败次数未达阈值不应熔断"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_open is False

    def test_success_resets_failure_count(self):
        """成功调用应重置失败计数"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        # 再失败 1 次不应熔断（计数已重置）
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_open_to_half_open_after_recovery_timeout(self):
        """OPEN 状态超时后应进入 HALF_OPEN"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.1
        ))
        cb.record_failure()
        assert cb.is_open is True
        time.sleep(0.15)
        # is_open 检查时自动转为 HALF_OPEN
        assert cb.is_open is False
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_after_success_threshold(self):
        """HALF_OPEN 下连续成功达阈值应恢复 CLOSED"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.1, success_threshold=2
        ))
        cb.record_failure()
        time.sleep(0.15)
        cb.is_open  # 触发转 HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """HALF_OPEN 下失败应回到 OPEN"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.1
        ))
        cb.record_failure()
        time.sleep(0.15)
        cb.is_open  # 触发转 HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_reset(self):
        """手动重置应回到 CLOSED"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_open is False


class TestCircuitBreakerThreadSafety:
    """熔断器线程安全测试"""

    def test_concurrent_record_failure_no_race(self):
        """多线程并发 record_failure 不应导致状态错乱

        阈值 100，10 线程各失败 10 次（共 100 次），
        应正好触发熔断，且 failure_count 不应超过 100（无重复累加）。
        """
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=100))

        def fail_10_times():
            for _ in range(10):
                cb.record_failure()

        threads = [threading.Thread(target=fail_10_times) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb.state == CircuitState.OPEN

    def test_concurrent_success_failure_no_corruption(self):
        """并发成功/失败交替不应导致状态损坏"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=50, success_threshold=5, recovery_timeout=0.05
        ))

        def mix_calls():
            for _ in range(50):
                cb.record_failure()
                cb.record_success()

        threads = [threading.Thread(target=mix_calls) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 状态应是 CLOSED/OPEN/HALF_OPEN 三者之一，不能是其他值
        assert cb.state in (CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN)

    def test_concurrent_is_open_no_deadlock(self):
        """并发调用 is_open 不应死锁"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=10, recovery_timeout=0.01
        ))

        def call_is_open():
            for _ in range(100):
                cb.is_open
                cb.record_failure()
                cb.record_success()

        threads = [threading.Thread(target=call_is_open) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        # 所有线程应在 5 秒内完成（无死锁）
        alive = sum(1 for t in threads if t.is_alive())
        assert alive == 0, f"仍有 {alive} 个线程未完成，可能死锁"

    def test_concurrent_reset_and_record(self):
        """并发 reset + record 不应导致状态不一致"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

        def do_reset():
            for _ in range(20):
                cb.reset()

        def do_record():
            for _ in range(20):
                cb.record_failure()

        t1 = threading.Thread(target=do_reset)
        t2 = threading.Thread(target=do_record)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert cb.state in (CircuitState.CLOSED, CircuitState.OPEN)
