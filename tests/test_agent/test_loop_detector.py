"""
LoopDetector 单元测试
===================
验证循环检测器的签名生成、滑动窗口、阈值触发和历史重建逻辑。
"""
from __future__ import annotations

import pytest

from backend.agent.loop_detector import LoopDetector


class TestLoopDetectorCore:
    def test_initial_state(self):
        """新检测器应保持空历史"""
        detector = LoopDetector(window_size=5, threshold=3)
        assert detector.history == []
        assert detector.is_loop("any", {}) is False

    def test_record_appends_to_history(self):
        """record 应向历史中添加签名"""
        detector = LoopDetector(window_size=5, threshold=3)
        detector.record("search", {"q": "timeout"})
        assert len(detector.history) == 1

    def test_window_size_evicts_old_records(self):
        """超过窗口大小时应丢弃最旧记录"""
        detector = LoopDetector(window_size=3, threshold=3)
        for i in range(5):
            detector.record("tool", {"i": i})
        assert len(detector.history) == 3
        assert detector.history[0] == LoopDetector._signature("tool", {"i": 2})

    def test_signature_is_stable(self):
        """相同工具名和参数应生成相同签名，不同顺序应一致"""
        sig1 = LoopDetector._signature("tool", {"a": 1, "b": 2})
        sig2 = LoopDetector._signature("tool", {"b": 2, "a": 1})
        sig3 = LoopDetector._signature("tool", {"a": 1, "b": 3})
        assert sig1 == sig2
        assert sig1 != sig3


class TestLoopDetectorIsLoop:
    def test_loop_triggered_at_threshold(self):
        """相同调用达到阈值时应判定为循环"""
        detector = LoopDetector(window_size=5, threshold=3)
        detector.record("search", {"q": "timeout"})
        detector.record("search", {"q": "timeout"})
        assert detector.is_loop("search", {"q": "timeout"}) is True

    def test_loop_not_triggered_below_threshold(self):
        """相同调用未达阈值时不应判定为循环"""
        detector = LoopDetector(window_size=5, threshold=3)
        detector.record("search", {"q": "timeout"})
        assert detector.is_loop("search", {"q": "timeout"}) is False

    def test_different_args_do_not_trigger_loop(self):
        """相同工具但不同参数不应判定为循环"""
        detector = LoopDetector(window_size=5, threshold=3)
        detector.record("search", {"q": "timeout"})
        detector.record("search", {"q": "timeout"})
        assert detector.is_loop("search", {"q": "login"}) is False

    def test_only_recent_window_counts(self):
        """窗口外的旧记录不应计入循环判断"""
        detector = LoopDetector(window_size=3, threshold=3)
        detector.record("search", {"q": "timeout"})
        detector.record("other", {"x": 1})
        detector.record("other", {"x": 2})
        # 窗口内只剩 1 条 search 记录
        assert detector.is_loop("search", {"q": "timeout"}) is False


class TestLoopDetectorCheckToolCalls:
    def test_detects_loop_in_tool_call_list(self):
        """批量检查应能识别循环"""
        detector = LoopDetector(window_size=5, threshold=2)
        detector.record("search", {"q": "timeout"})
        tool_calls = [{"name": "search", "args": {"q": "timeout"}}]
        is_loop, msg = detector.check_tool_calls(tool_calls)
        assert is_loop is True
        assert "search" in msg

    def test_no_loop_when_tools_differ(self):
        """不同工具调用不应被误判为循环"""
        detector = LoopDetector(window_size=5, threshold=2)
        tool_calls = [
            {"name": "search", "args": {"q": "timeout"}},
            {"name": "click", "args": {"id": "btn"}},
        ]
        is_loop, msg = detector.check_tool_calls(tool_calls)
        assert is_loop is False
        assert msg == ""


class TestLoopDetectorFromHistory:
    def test_reconstruct_from_history(self):
        """from_history 应从历史记录重建检测器状态"""
        history = [
            {"tool_name": "search", "tool_args": {"q": "a"}},
            {"tool_name": "search", "tool_args": {"q": "a"}},
        ]
        detector = LoopDetector.from_history(history, window_size=5, threshold=3)
        assert detector.is_loop("search", {"q": "a"}) is True

    def test_from_history_respects_window_size(self):
        """from_history 应只加载窗口大小范围内的记录"""
        history = [
            {"tool_name": "search", "tool_args": {"q": "a"}},
            {"tool_name": "search", "tool_args": {"q": "a"}},
            {"tool_name": "other", "tool_args": {}},
            {"tool_name": "other", "tool_args": {}},
        ]
        detector = LoopDetector.from_history(history, window_size=2, threshold=2)
        # 窗口内为两条 other，is_loop 对 other 返回 True
        assert detector.is_loop("other", {}) is True
        # search 已滑出窗口
        assert detector.is_loop("search", {"q": "a"}) is False

    def test_from_history_ignores_invalid_items(self):
        """from_history 对异常项应能容错"""
        history = [
            {"tool_name": "search", "tool_args": {"q": "a"}},
            {"tool_name": "search"},  # 缺少 tool_args
        ]
        detector = LoopDetector.from_history(history, window_size=5, threshold=3)
        # 只有一条有效记录，不会触发循环
        assert detector.is_loop("search", {"q": "a"}) is False
