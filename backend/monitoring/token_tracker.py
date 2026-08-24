"""
LLM Token 消耗追踪
==================

进程内聚合每次 LLM 调用的 token 用量，并同时上报 Prometheus 指标
（llm_tokens_total）。数据来源为 Provider 层解析的响应 usage 字段——
只记录真实值，不做估算：
- chat 非流式 / chat_with_tools / embed：记录 API 返回的真实 usage
- chat 流式：OpenAI 兼容接口流式响应不含 usage，不计入（调用次数
  仍由 llm_calls_total 覆盖）

查询方式：
- Prometheus: llm_tokens_total{provider, operation, type}
- JSON API: GET /api/v1/metrics/tokens（TokenUsageTracker.snapshot()）
"""
import logging
import threading

from backend.monitoring.metrics import LLM_TOKENS_TOTAL

logger = logging.getLogger("ai_rd_agent")


class TokenUsageTracker:
    """进程级 Token 用量聚合器（线程安全）"""

    def __init__(self):
        self._lock = threading.Lock()
        # key: f"{provider}:{operation}" → [calls, prompt_tokens, completion_tokens]
        self._usage: dict[str, list[int]] = {}

    def record(
        self,
        provider: str,
        operation: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """记录一次 LLM 调用的 token 用量

        数值防御：非正数跳过（Provider 层的 MagicMock usage 会被此条件过滤，
        避免测试桩污染指标）。
        """
        if prompt_tokens <= 0 and completion_tokens <= 0:
            return

        with self._lock:
            key = f"{provider}:{operation}"
            stats = self._usage.setdefault(key, [0, 0, 0])
            stats[0] += 1
            stats[1] += prompt_tokens
            stats[2] += completion_tokens

        if prompt_tokens > 0:
            LLM_TOKENS_TOTAL.labels(provider, operation, "prompt").inc(prompt_tokens)
        if completion_tokens > 0:
            LLM_TOKENS_TOTAL.labels(provider, operation, "completion").inc(completion_tokens)

        logger.info(
            f"Token 用量 | {provider}.{operation} | "
            f"prompt={prompt_tokens} completion={completion_tokens}"
        )

    def snapshot(self) -> dict:
        """返回当前聚合快照（JSON 可序列化）"""
        with self._lock:
            items = []
            total_calls = total_prompt = total_completion = 0
            for key, (calls, pt, ct) in sorted(self._usage.items()):
                provider, _, operation = key.partition(":")
                items.append({
                    "provider": provider,
                    "operation": operation,
                    "calls": calls,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                })
                total_calls += calls
                total_prompt += pt
                total_completion += ct

            return {
                "totals": {
                    "calls": total_calls,
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total_prompt + total_completion,
                },
                "by_operation": items,
            }

    def reset(self) -> None:
        """清空聚合数据（仅测试使用）"""
        with self._lock:
            self._usage.clear()


# 进程级单例：Provider 层埋点统一写入此实例
token_tracker = TokenUsageTracker()
