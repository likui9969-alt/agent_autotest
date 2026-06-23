"""
日志分析路由
POST /api/v1/analysis/log          — 上传日志文件或粘贴日志内容，返回故障分析报告
GET  /api/v1/analysis/runtime-logs — 读取后端服务最近的运行日志（无需上传文件）
"""
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse

from backend.models.analysis import LogAnalysisRequest, AnalysisResult
from backend.api.deps import get_log_analyzer

logger = logging.getLogger("ai_rd_agent")
router = APIRouter(tags=["日志分析"])


@router.post("/log", response_model=AnalysisResult)
async def analyze_log(
    log_content: str = Form(default="", description="日志文本内容（直接粘贴）"),
    file: UploadFile = File(default=None, description="日志文件（.log/.txt）"),
    include_historical: bool = Form(default=True, description="是否检索历史案例"),
    top_k: int = Form(default=3, description="检索历史案例数量"),
):
    """分析测试日志并生成故障分析报告

    支持两种输入方式：
    1. 直接在 log_content 中粘贴日志文本
    2. 通过 file 上传 .log / .txt 文件

    也支持同时提供，优先使用上传文件的内容。
    """
    # ---- 确定日志来源 ----
    content = log_content
    filename = "manual_input.log"

    if file and file.filename:
        # 从上传文件中读取（一次性读取全部字节，避免重复 read 返回空）
        raw_bytes = await file.read()
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = raw_bytes.decode("gbk", errors="replace")
        filename = file.filename

    if not content.strip():
        return JSONResponse(
            status_code=400,
            content={"error": True, "message": "请提供日志内容（粘贴或上传文件）"},
        )

    # ---- 执行分析 ----
    request = LogAnalysisRequest(
        log_content=content,
        filename=filename,
        include_historical=include_historical,
        top_k=top_k,
    )

    analyzer = get_log_analyzer()
    result = analyzer.analyze(request)

    return result


@router.get("/runtime-logs")
async def get_runtime_logs(
    tail_lines: int = Query(default=200, ge=1, le=2000, description="读取最后 N 行日志"),
    level: str = Query(
        default="all",
        description="日志级别过滤：all / ERROR / WARNING / INFO",
    ),
):
    """读取后端服务最近的运行日志（无需上传文件）

    用于"刚才发生了什么错误"这类场景：直接拉取 data/logs/app.log 的尾部内容。
    """
    from backend.config.settings import get_settings

    settings = get_settings()
    log_file = Path(settings.get_log_dir()) / "app.log"

    if not log_file.exists():
        return JSONResponse(
            status_code=404,
            content={"error": True, "message": "运行日志文件不存在（服务可能刚启动）"},
        )

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": True, "message": f"读取日志失败: {e}"},
        )

    level_upper = level.upper()
    if level_upper in ("ERROR", "WARNING", "INFO"):
        filtered = [ln for ln in all_lines if f"| {level_upper}" in ln]
    else:
        filtered = all_lines

    tail = filtered[-tail_lines:]

    return {
        "filename": str(log_file),
        "level": level_upper,
        "total_lines": len(all_lines),
        "filtered_lines": len(filtered),
        "returned_lines": len(tail),
        "content": "".join(tail),
    }
