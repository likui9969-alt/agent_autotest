"""知识库管理页。"""
from __future__ import annotations

import streamlit as st

from utils.api import api_get, api_post, get_api_host


def render() -> None:
    API_BASE = get_api_host()

    st.title("📚 知识库管理")
    st.markdown("管理测试知识库文档，支持上传、查看和重建向量库。")

    col1, col2 = st.columns([2, 1])

    with col1:
        # ---- 文档上传 ----
        st.subheader("上传文档")
        uploaded_file = st.file_uploader(
            "选择文档文件",
            type=["txt", "pdf", "docx"],
            help="支持 txt、pdf、docx 格式",
        )

        if uploaded_file:
            if st.button("🚀 上传并索引", type="primary"):
                with st.spinner("正在处理文档..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    success, data = api_post(
                        "api/v1/knowledge/upload",
                        files=files,
                        timeout=120,
                    )
                    if success:
                        st.success(f"✅ 文档 '{data['filename']}' 上传成功！")
                        st.info(f"切分为 {data['chunk_count']} 个文本块")
                        st.rerun()
                    else:
                        st.error(f"上传失败: {data}")

        # ---- 重建向量库 ----
        st.subheader("重建向量库")
        st.caption("清空所有已有向量，重新索引 data/docs/ 目录下的所有文档")
        if st.button("🔄 重建向量库", type="secondary"):
            with st.spinner("正在重建向量库..."):
                success, data = api_post("api/v1/knowledge/rebuild", timeout=300)
                if success:
                    st.success(data.get("message", "重建完成"))
                    st.rerun()
                else:
                    st.error(f"重建失败: {data}")

    with col2:
        # ---- 知识库统计 ----
        st.subheader("知识库概况")
        if st.button("🔄 刷新统计"):
            st.rerun()

        success, stats = api_get("api/v1/knowledge/stats", timeout=10)
        if success:
            st.metric("文档总数", stats.get("total_documents", 0))
            st.metric("向量块数", stats.get("total_chunks", 0))
            st.metric("集合名称", stats.get("collection_name", "N/A"))
        else:
            st.warning("无法获取统计信息")
            if "无法连接" in str(stats):
                st.warning(
                    f"⚠ 后端服务未连接\n\n请先启动后端：\n```\nuvicorn backend.main:app --reload\n```"
                )


if __name__ == "__main__":
    render()
