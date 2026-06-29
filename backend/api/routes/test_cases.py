"""
测试用例管理路由
==================
提供测试用例自动生成、列表、详情、删除、导出接口。
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.agent.test_generator import TestCaseGenerator
from backend.api.deps import get_llm_client
from backend.db.test_cases import TestCaseStore
from backend.models.test_case import (
    TestCase,
    TestCaseGenerateRequest,
    TestCaseGenerateResponse,
)

logger = logging.getLogger("ai_rd_agent")
router = APIRouter(tags=["测试用例管理"])

# 全局单例
_test_case_store: Optional[TestCaseStore] = None


def _get_store() -> TestCaseStore:
    global _test_case_store
    if _test_case_store is None:
        _test_case_store = TestCaseStore()
    return _test_case_store


@router.post("/generate", response_model=TestCaseGenerateResponse)
async def generate_test_cases(request: TestCaseGenerateRequest):
    """根据需求自动生成测试用例"""
    try:
        generator = TestCaseGenerator(llm_client=get_llm_client())
        cases = generator.generate(request)
        _get_store().save_many(cases)
        return TestCaseGenerateResponse(
            status="success",
            generated_count=len(cases),
            test_cases=cases,
            message=f"成功生成并保存 {len(cases)} 条测试用例",
        )
    except Exception as e:
        logger.error(f"测试用例生成失败: {e}", exc_info=True)
        return TestCaseGenerateResponse(
            status="failed",
            generated_count=0,
            message=f"生成失败: {str(e)}",
        )


@router.get("/")
async def list_test_cases(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """获取测试用例列表"""
    store = _get_store()
    cases = store.list_cases(limit=limit, offset=offset)
    return {
        "total": store.count(),
        "limit": limit,
        "offset": offset,
        "test_cases": [case.model_dump() for case in cases],
    }


@router.get("/{case_id}")
async def get_test_case(case_id: str):
    """获取单条测试用例详情"""
    case = _get_store().get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="用例不存在")
    return case.model_dump()


@router.delete("/{case_id}")
async def delete_test_case(case_id: str):
    """删除测试用例"""
    success = _get_store().delete_case(case_id)
    if not success:
        raise HTTPException(status_code=404, detail="用例不存在")
    return {"status": "success", "message": "已删除"}


@router.post("/{case_id}/export")
async def export_test_case(
    case_id: str,
    format: str = Query("json", regex="^(json|csv|excel)$"),
):
    """导出单条测试用例"""
    case = _get_store().get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="用例不存在")

    if format == "json":
        data = json.dumps(case.model_dump(), ensure_ascii=False, indent=2)
        media_type = "application/json"
        filename = f"testcase_{case_id}.json"
    elif format == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=case.to_row().keys())
        writer.writeheader()
        writer.writerow(case.to_row())
        data = buffer.getvalue()
        media_type = "text/csv; charset=utf-8"
        filename = f"testcase_{case_id}.csv"
    else:
        try:
            import pandas as pd
            df = pd.DataFrame([case.to_row()])
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine="openpyxl")
            data = buffer.getvalue()
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"testcase_{case_id}.xlsx"
        except ImportError:
            raise HTTPException(status_code=500, detail="缺少 pandas/openpyxl，无法导出 Excel")

    return StreamingResponse(
        io.BytesIO(data.encode("utf-8") if isinstance(data, str) else data),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/export/bulk")
async def export_all_test_cases(
    format: str = Query("csv", regex="^(csv|excel|json)$"),
):
    """批量导出所有测试用例"""
    cases = _get_store().list_cases(limit=10000)
    if not cases:
        raise HTTPException(status_code=404, detail="没有用例可导出")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if format == "json":
        data = json.dumps([c.model_dump() for c in cases], ensure_ascii=False, indent=2)
        media_type = "application/json"
        filename = f"testcases_{timestamp}.json"
        payload = data.encode("utf-8")
    elif format == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=cases[0].to_row().keys())
        writer.writeheader()
        for case in cases:
            writer.writerow(case.to_row())
        payload = buffer.getvalue().encode("utf-8-sig")
        media_type = "text/csv; charset=utf-8"
        filename = f"testcases_{timestamp}.csv"
    else:
        try:
            import pandas as pd
            df = pd.DataFrame([c.to_row() for c in cases])
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine="openpyxl")
            payload = buffer.getvalue()
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"testcases_{timestamp}.xlsx"
        except ImportError:
            raise HTTPException(status_code=500, detail="缺少 pandas/openpyxl，无法导出 Excel")

    return StreamingResponse(
        io.BytesIO(payload),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
