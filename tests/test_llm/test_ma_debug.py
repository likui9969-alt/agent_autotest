"""CI 诊断测试（临时）

定位 GitHub Actions 上 LLMClient() 构造失败的根因：
LLMClient.__init__ 会吞掉 provider 初始化异常（logger.warning），
此测试捕获该日志并塞进断言消息，经 annotate-failures 插件
上报到 GitHub annotations 便于远程定位。

文件名以 zz 开头确保在全量测试的最后执行（此时全局状态已就位）。
"""


def test_zz_diagnose_llm_init():
    import io
    import logging

    from backend.config.settings import get_settings

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("ai_rd_agent")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    settings = get_settings()
    env_snapshot = {
        "DASHSCOPE_API_KEY(set?": bool(settings.DASHSCOPE_API_KEY),
        "prefix": repr(settings.DASHSCOPE_API_KEY[:8]),
        "DASHSCOPE_URL": repr(settings.DASHSCOPE_URL[:40]),
        "LLM_MODEL": repr(settings.LLM_MODEL),
        "LLM_PROVIDER": repr(settings.LLM_PROVIDER),
        "LLM_PROVIDERS": repr(settings.LLM_PROVIDERS),
        "EMBEDDING_MODEL": repr(settings.EMBEDDING_MODEL),
    }

    try:
        from backend.llm.client import LLMClient

        LLMClient()
    except ValueError as e:
        raise AssertionError(
            f"LLMClient init FAILED: {e}\n"
            f"settings snapshot: {env_snapshot}\n"
            f"captured provider logs:\n{stream.getvalue()}"
        ) from e
    finally:
        logger.removeHandler(handler)

    # 若成功，也输出快照供对照
    print(f"LLMClient OK. settings: {env_snapshot}")
