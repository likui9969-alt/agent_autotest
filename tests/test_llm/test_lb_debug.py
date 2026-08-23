"""CI 诊断测试（临时）

定位 GitHub Actions 上 LLMClient() 构造失败的根因。
LLMClient.__init__ 会吞掉 provider 初始化异常（logger.warning），
此测试捕获该日志与 settings 快照塞进断言消息，经
annotate-failures 插件上报到 GitHub annotations 便于远程定位。

文件名 lb 排序位于 test_llm.py 之前（污染窗口内）。
"""


def test_lb_diagnose_llm_init():
    import io
    import logging
    import os
    import traceback

    from backend.config.settings import get_settings

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("ai_rd_agent")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    settings = get_settings()
    env_snapshot = {
        "settings.DASHSCOPE_API_KEY(bool)": bool(settings.DASHSCOPE_API_KEY),
        "settings.DASHSCOPE_API_KEY[:8]": repr(settings.DASHSCOPE_API_KEY[:8]),
        "settings.DASHSCOPE_URL": repr(settings.DASHSCOPE_URL[:40]),
        "settings.LLM_MODEL": repr(settings.LLM_MODEL),
        "settings.LLM_PROVIDER": repr(settings.LLM_PROVIDER),
        "settings.LLM_PROVIDERS": repr(settings.LLM_PROVIDERS),
        "settings.EMBEDDING_MODEL": repr(settings.EMBEDDING_MODEL),
        "os.environ DASHSCOPE_API_KEY(bool)": bool(os.environ.get("DASHSCOPE_API_KEY")),
        "os.environ DASHSCOPE_API_KEY[:8]": repr((os.environ.get("DASHSCOPE_API_KEY") or "")[:8]),
        "settings type": type(settings).__name__,
        "lru_cache info": repr(get_settings.cache_info()),
    }

    # 逐项尝试 DashScopeProvider 构造（不吞异常）
    provider_error = ""
    try:
        from backend.llm.providers.dashscope import DashScopeProvider

        DashScopeProvider()
    except Exception:
        provider_error = traceback.format_exc()

    try:
        from backend.llm.client import LLMClient

        LLMClient()
        init_result = "OK"
    except ValueError as e:
        init_result = f"FAILED: {e}"

    # 临时诊断：始终上报快照（无论成败），便于比对 CI 与本地差异
    raise AssertionError(
        f"LLMClient init: {init_result}\n"
        f"DashScopeProvider direct error:\n{provider_error or '(none)'}\n"
        f"snapshot: {env_snapshot}\n"
        f"captured logs:\n{stream.getvalue() or '(none)'}"
    )
