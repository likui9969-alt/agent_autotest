"""智能问答页。"""
from __future__ import annotations

import streamlit as st

from utils.api import api_post, get_api_host


def render() -> None:
    API_BASE = get_api_host()

    st.title("💬 智能问答")
    st.markdown("基于知识库的 RAG 检索增强生成问答。输入测试相关问题，获取 AI 驱动的分析回答。")

    # 检索参数
    with st.expander("⚙ 检索设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("检索文档数", min_value=1, max_value=20, value=5)
        with col2:
            search_type = st.selectbox(
                "检索方式",
                options=["similarity", "mmr"],
                format_func=lambda x: "相似度检索" if x == "similarity" else "MMR 检索",
                help="相似度检索返回最相关文档；MMR 在相关性基础上增加多样性",
            )
        include_sources = st.checkbox("显示引用来源", value=True)

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史对话
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📎 引用来源"):
                    for src in msg["sources"]:
                        st.caption(f"**{src.get('source_file', '未知')}** (相关度: {src.get('score', 0):.3f})")
                        st.text(src.get("excerpt", "")[:300])

    # 输入框
    if question := st.chat_input("请输入你的问题，例如：登录接口返回500错误怎么办？"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 调用 RAG API
        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并生成回答..."):
                success, data = api_post(
                    "api/v1/rag/query",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "search_type": search_type,
                        "include_sources": include_sources,
                    },
                    timeout=120,
                )
                if success:
                    answer = data.get("answer", "无法获取回答")
                    sources = data.get("sources", [])
                    elapsed = data.get("response_time_ms", 0)

                    st.markdown(answer)
                    st.caption(f"⏱ 耗时 {elapsed:.0f}ms | 检索到 {data.get('retrieved_count', 0)} 个文档")

                    if include_sources and sources:
                        with st.expander("📎 引用来源"):
                            for src in sources:
                                st.caption(
                                    f"**{src.get('source_file', '未知')}** "
                                    f"(相关度: {src.get('score', 0):.3f})"
                                )
                                st.text(src.get("excerpt", "")[:300])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    st.error(f"查询失败: {data}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ 查询失败: {data}",
                        "sources": [],
                    })

    # 清空对话按钮
    if st.session_state.messages and st.button("🗑 清空对话", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    render()
