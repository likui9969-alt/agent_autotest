"""历史报告页。"""
from __future__ import annotations

import streamlit as st

from utils.api import api_get, get_api_host


def render() -> None:
    API_BASE = get_api_host()

    st.title("📋 历史报告")
    st.markdown("查看已持久化的自动化测试和 Agent 执行记录。")

    report_type = st.selectbox(
        "报告类型",
        ["全部", "自动化测试", "Agent 执行"],
        key="report_type_filter",
    )

    type_param = {
        "全部": "all",
        "自动化测试": "test",
        "Agent 执行": "agent",
    }[report_type]

    def _fetch_reports(endpoint: str, limit: int = 50) -> list:
        success, data = api_get(
            f"api/v1/{endpoint}/reports",
            params={"limit": limit},
            timeout=10,
        )
        if success:
            return data.get("reports", [])
        return []

    if type_param == "all":
        test_reports = _fetch_reports("testing")
        agent_reports = _fetch_reports("agent")
        reports = test_reports + agent_reports
        # 按时间倒序排列
        reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif type_param == "agent":
        reports = _fetch_reports("agent")
    else:
        reports = _fetch_reports("testing")

    if not reports:
        st.info("暂无历史报告，请执行测试或 Agent 任务后查看。")
    else:
        st.write(f"共 {len(reports)} 条记录")
        for r in reports:
            with st.expander(
                f"[{r.get('report_type', 'unknown').upper()}] "
                f"{r.get('name', '未命名')} — "
                f"{r.get('status', 'unknown')} — "
                f"{r.get('created_at', '')[:19]}"
            ):
                st.write(f"**ID**: `{r.get('id', '')}`")
                st.write(f"**状态**: {r.get('status', '')}")
                st.write(f"**时间**: {r.get('created_at', '')}")
                if st.button("查看详情", key=f"report_detail_{r.get('id', '')}"):
                    detail_path = f"api/v1/{'agent' if r.get('report_type') == 'agent' else 'testing'}/reports/{r.get('id', '')}"
                    success, detail = api_get(detail_path, timeout=10)
                    if success:
                        st.json(detail)
                    else:
                        st.error(f"获取详情失败: {detail}")


if __name__ == "__main__":
    render()
