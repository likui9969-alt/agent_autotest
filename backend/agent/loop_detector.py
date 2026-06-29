"""
Agent 循环检测器

检测 ReAct 循环中 LLM 是否重复调用相同工具且参数相同，
防止 Agent 陷入死循环消耗 token。
"""
import json
import logging

logger = logging.getLogger("ai_rd_agent")


class LoopDetector:
    """工具调用循环检测器

    维护最近 N 次工具调用签名，当相同签名的调用连续达到阈值时，
    判定为循环并建议终止。

    使用示例：
        detector = LoopDetector(window_size=5, threshold=3)
        detector.record("search_knowledge_base", {"query": "timeout"})
        if detector.is_loop("search_knowledge_base", {"query": "timeout"}):
            print("检测到循环")
    """

    def __init__(self, window_size: int = 5, threshold: int = 3):
        """
        Args:
            window_size: 滑动窗口大小，记录最近多少次工具调用
            threshold: 触发循环的相同调用次数阈值（含当前这次）
        """
        self.window_size = window_size
        self.threshold = threshold
        self.history: list[str] = []

    @staticmethod
    def _signature(tool_name: str, tool_args: dict) -> str:
        """生成工具调用签名，用于比较两次调用是否相同"""
        try:
            args_str = json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(tool_args)
        return f"{tool_name}:{args_str}"

    def record(self, tool_name: str, tool_args: dict) -> None:
        """记录一次工具调用"""
        signature = self._signature(tool_name, tool_args)
        self.history.append(signature)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def is_loop(self, tool_name: str, tool_args: dict) -> bool:
        """检查当前这次调用是否会触发循环

        检查逻辑：当前这次调用 + 历史窗口中相同签名的调用次数 >= threshold
        """
        signature = self._signature(tool_name, tool_args)
        recent_same = sum(
            1 for s in self.history[-self.window_size :]
            if s == signature
        )
        return recent_same + 1 >= self.threshold

    def check_tool_calls(self, tool_calls: list[dict]) -> tuple[bool, str]:
        """批量检查一次 LLM 返回的工具调用列表

        Returns:
            (是否检测到循环, 原因说明)
        """
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            if self.is_loop(name, args):
                signature = self._signature(name, args)
                msg = (
                    f"检测到 Agent 循环：工具 `{name}` 以相同参数连续调用 "
                    f"达到阈值 {self.threshold} 次（签名：{signature}），已自动终止。"
                )
                logger.warning(f"[LoopDetector] {msg}")
                return True, msg
            # 先临时记录，用于后续工具调用的循环判断
            self.record(name, args)
        return False, ""

    @classmethod
    def from_history(
        cls,
        history: list[dict],
        window_size: int = 5,
        threshold: int = 3,
    ) -> "LoopDetector":
        """从历史记录重建检测器"""
        detector = cls(window_size=window_size, threshold=threshold)
        for item in history[-window_size:]:
            detector.record(item.get("tool_name", ""), item.get("tool_args", {}))
        return detector
