"""
健康检查路由
GET /health — 返回服务及依赖组件的存活状态
GET /system/selenium-diagnose — 返回 Chrome / chromedriver 诊断信息
"""
import logging
from fastapi import APIRouter

logger = logging.getLogger("ai_rd_agent")

router = APIRouter(tags=["系统"])


@router.get("/health/lite")
async def health_check_lite():
    """轻量级健康检查 — 仅返回进程存活，不调 LLM/Chroma，用于前端探测"""
    return {"status": "ok", "service": "AI研发效能智能体"}


@router.get("/health")
async def health_check():
    """健康检查接口 — 验证服务和关键依赖是否正常"""
    health_status = {
        "status": "ok",
        "service": "AI研发效能智能体",
    }

    try:
        from backend.config.settings import get_settings
        settings = get_settings()
        health_status["llm_model"] = settings.LLM_MODEL
        health_status["embedding_model"] = settings.EMBEDDING_MODEL
    except Exception as e:
        logger.warning(f"健康检查读取配置失败: {e}")
        health_status["status"] = "error"
        health_status["errors"] = [f"config: {e}"]
        return health_status

    # 检测 LLM 可用性（轻量嵌入调用）
    try:
        from backend.api.deps import get_llm_client
        llm_client = get_llm_client()
        llm_client.embed_single("health_check")
        health_status["llm"] = "ok"
    except Exception as e:
        logger.warning(f"LLM 健康检查失败: {e}")
        health_status["llm"] = "error"
        health_status.setdefault("errors", []).append(f"llm: {e}")
        health_status["status"] = "degraded"

    # 检测 Chroma 可用性
    try:
        from backend.api.deps import get_rag_pipeline
        pipeline = get_rag_pipeline()
        count = pipeline.vector_store.count()
        health_status["chroma"] = "ok"
        health_status["vector_chunks"] = count
    except Exception as e:
        logger.warning(f"Chroma 健康检查失败: {e}")
        health_status["chroma"] = "error"
        health_status.setdefault("errors", []).append(f"chroma: {e}")
        health_status["status"] = "degraded"

    return health_status


@router.get("/system/selenium-diagnose")
async def selenium_diagnose():
    """诊断 Selenium 环境：Chrome 浏览器、chromedriver、版本匹配情况

    返回结构化诊断信息，帮助用户快速定位真实浏览器测试无法启动的原因。
    """
    from pathlib import Path
    from backend.config.settings import get_settings
    from backend.selenium_driver.driver import detect_chrome
    from backend.selenium_driver.driver import (
        _get_chromedriver_major_version,
        _get_chrome_major_version,
    )

    settings = get_settings()
    chrome_binary, driver_path = detect_chrome()

    chrome_version = _get_chrome_major_version(chrome_binary) if chrome_binary else None
    driver_version = _get_chromedriver_major_version(driver_path) if driver_path else None

    # 判断各组件状态
    chrome_found = bool(chrome_binary) and Path(chrome_binary).exists()
    driver_found = bool(driver_path) and Path(driver_path).exists()

    version_match = None
    if chrome_version and driver_version:
        version_match = chrome_version == driver_version

    # 汇总错误原因/提示
    errors = []
    tips = []
    if not chrome_found:
        errors.append("未找到 Chrome 浏览器，请确认已安装 Google Chrome")
    if not driver_found:
        tips.append("未找到本地 chromedriver，真实浏览器测试时将尝试让 Selenium 自动下载（需联网）")
    if chrome_found and driver_found and version_match is False:
        tips.append(
            f"Chrome (v{chrome_version}) 与本地 chromedriver (v{driver_version}) 主版本不匹配，"
            f"系统将尝试使用缓存中的 v{chrome_version} 驱动或让 Selenium 自动下载匹配版本"
        )

    # 即使本地驱动不匹配，只要 Chrome 存在且允许 Selenium 自动管理，就视为可用
    ready = chrome_found and (
        version_match is True or version_match is None or bool(tips)
    )

    return {
        "ready": ready,
        "chrome_found": chrome_found,
        "chrome_binary": chrome_binary or "",
        "chrome_version": chrome_version,
        "chromedriver_found": driver_found,
        "chromedriver_path": driver_path or "",
        "chromedriver_version": driver_version,
        "version_match": version_match,
        "configured_chromedriver_path": settings.CHROMEDRIVER_PATH,
        "configured_chrome_binary_path": settings.CHROME_BINARY_PATH,
        "errors": errors,
        "tips": tips,
        "message": "环境就绪" if ready else "；".join(errors) if errors else "；".join(tips) if tips else "检测中",
    }
