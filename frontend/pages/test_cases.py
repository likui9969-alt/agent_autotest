"""测试用例自动生成页。"""
from __future__ import annotations

import streamlit as st

from utils.api import api_get, api_post, get_api_host


def render() -> None:
    API_BASE = get_api_host()
    st.title("📝 测试用例生成")
    st.markdown("基于 LLM + RAG 知识库，根据需求描述自动生成结构化测试用例。")

    tab_gen, tab_list = st.tabs(["生成用例", "用例库"])

    with tab_gen:
        requirement = st.text_area(
            "需求描述",
            height=120,
            placeholder="例如：电商 App 用户登录功能，需要支持手机号/密码登录。",
        )
        scenario = st.text_input("补充场景说明（可选）", "")
        count = st.slider("生成数量", 1, 20, 5)
        use_knowledge = st.checkbox("检索知识库补充历史用例", value=True)

        if st.button("✨ 生成测试用例", type="primary", disabled=not requirement.strip()):
            with st.spinner("AI 正在生成用例..."):
                success, data = api_post(
                    "api/v1/test-cases/generate",
                    json={
                        "requirement": requirement.strip(),
                        "scenario": scenario,
                        "count": count,
                        "use_knowledge": use_knowledge,
                    },
                    timeout=120,
                )
                if success and data.get("status") == "success":
                    st.success(data.get("message", f"生成 {data.get('generated_count', 0)} 条用例"))
                    cases = data.get("test_cases", [])
                    for case in cases:
                        with st.expander(f"[{case.get('priority', '中')}] {case.get('title', '')}"):
                            st.markdown(f"**模块：** {case.get('module', '')}")
                            st.markdown(f"**目标：** {case.get('objective', '')}")
                            st.markdown(f"**前置条件：** {case.get('preconditions', '')}")
                            st.markdown("**步骤：**")
                            for i, step in enumerate(case.get("steps", []), 1):
                                st.markdown(f"{i}. {step}")
                            st.markdown(f"**预期结果：** {case.get('expected_result', '')}")
                else:
                    st.error(f"生成失败：{data.get('message', data)}")

    with tab_list:
        st.subheader("已保存用例")
        success, data = api_get("api/v1/test-cases/", timeout=10)
        if not success:
            st.error(f"获取用例失败：{data}")
            return

        cases = data.get("test_cases", [])
        total = data.get("total", 0)
        st.caption(f"共 {total} 条用例")

        col1, col2 = st.columns([1, 3])
        with col1:
            export_format = st.selectbox("批量导出格式", ["csv", "excel", "json"], key="tc_bulk_format")
        with col2:
            if st.button("📥 批量导出", key="tc_bulk_export"):
                url = f"{API_BASE}/api/v1/test-cases/export/bulk?format={export_format}"
                import requests
                try:
                    resp = requests.post(url, timeout=30)
                    if resp.status_code == 200:
                        ext = {"csv": "csv", "excel": "xlsx", "json": "json"}[export_format]
                        st.download_button(
                            label=f"下载 {ext.upper()}",
                            data=resp.content,
                            file_name=f"testcases_export.{ext}",
                            mime=resp.headers.get("Content-Type", "application/octet-stream"),
                            key="tc_download_bulk",
                        )
                    else:
                        st.error(f"导出失败：{resp.status_code}")
                except Exception as e:
                    st.error(f"导出异常：{e}")

        if not cases:
            st.info("暂无保存的用例")
            return

        for case in cases:
            with st.expander(f"[{case.get('priority', '中')}] {case.get('title', '')}"):
                st.markdown(f"**模块：** {case.get('module', '')}")
                st.markdown(f"**目标：** {case.get('objective', '')}")
                st.markdown("**步骤：**")
                for i, step in enumerate(case.get("steps", []), 1):
                    st.markdown(f"{i}. {step}")
                st.markdown(f"**预期结果：** {case.get('expected_result', '')}")

                c1, c2 = st.columns([1, 1])
                with c1:
                    fmt = st.selectbox("导出", ["csv", "excel", "json"], key=f"tc_fmt_{case['id']}")
                with c2:
                    if st.button("📥 导出", key=f"tc_export_{case['id']}"):
                        url = f"{API_BASE}/api/v1/test-cases/{case['id']}/export?format={fmt}"
                        try:
                            import requests
                            resp = requests.post(url, timeout=30)
                            if resp.status_code == 200:
                                ext = {"csv": "csv", "excel": "xlsx", "json": "json"}[fmt]
                                st.download_button(
                                    label="下载",
                                    data=resp.content,
                                    file_name=f"testcase_{case['id']}.{ext}",
                                    mime=resp.headers.get("Content-Type", "application/octet-stream"),
                                    key=f"tc_dl_{case['id']}",
                                )
                            else:
                                st.error(f"导出失败：{resp.status_code}")
                        except Exception as e:
                            st.error(f"导出异常：{e}")


if __name__ == "__main__":
    render()
