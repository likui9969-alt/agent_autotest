"""Streamlit 前端应用 — AI-Driven 研发效能智能体

启动命令：streamlit run frontend/app.py
"""
from __future__ import annotations

import streamlit as st

from utils.api import _DEFAULT_API_BASE, auto_detect_backend, probe_backend
from utils.state import init_state
from utils.style import GLOBAL_CSS

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AI研发效能智能体",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 全局样式优化 ====================
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ==================== 初始化状态 & API 配置 ====================
init_state()

# API 地址配置：优先从 URL query param 读取，其次自动探测，最后使用默认值
_initial_api_host = st.session_state.get("api_host", _DEFAULT_API_BASE)

query_api_host = st.query_params.get("api_host", "")
if query_api_host:
    _initial_api_host = str(query_api_host).rstrip("/")
elif st.session_state.get("api_host", _DEFAULT_API_BASE) == _DEFAULT_API_BASE:
    _detected = auto_detect_backend()
    if _detected != _DEFAULT_API_BASE:
        _initial_api_host = _detected

st.session_state.api_host = _initial_api_host

# ==================== 侧边栏 ====================
st.sidebar.title("🤖 AI研发效能智能体")
st.sidebar.markdown("基于 RAG 的自动化测试与故障分析系统")
st.sidebar.divider()

# 自动探测按钮 + 连接状态
_detect_col, _status_col = st.sidebar.columns([1, 1])
if _detect_col.button("🔍 自动探测", key="detect_btn", help="扫描 8000-8005 端口自动找到后端"):
    with st.spinner("扫描中..."):
        detected = auto_detect_backend()
        st.session_state.api_host = detected
        st.query_params["api_host"] = detected
        st.rerun()

api_host = st.sidebar.text_input(
    "后端API地址",
    value=st.session_state.api_host,
    key="api_host_input",
    help="修改后按回车生效，地址会同步到浏览器 URL 方便刷新保留",
)
if api_host:
    api_host = api_host.rstrip("/")
    st.session_state.api_host = api_host
    if st.query_params.get("api_host") != api_host:
        st.query_params["api_host"] = api_host

# 实时连接状态指示灯（在地址最终确定后探测，避免冷启动时误判）
_is_connected = probe_backend(st.session_state.api_host, timeout=2.0)
_status_emoji = "🟢" if _is_connected else "🔴"
_status_text = "已连接" if _is_connected else "未连接"
_status_col.markdown(f"{_status_emoji} **{_status_text}**")

_api_port = st.session_state.api_host.split(":")[-1] if ":" in st.session_state.api_host else "?"
st.sidebar.caption(f"当前端口: `{_api_port}` | 状态: {_status_text}")

st.sidebar.divider()

# JIRA 连接状态（设置入口）
from utils.api import api_get
st.sidebar.markdown("### 🎫 JIRA 状态")
if st.sidebar.button("测试 JIRA 连接", key="jira_check_btn"):
    jira_success, jira_data = api_get("api/v1/jira/status", timeout=10)
    if jira_success:
        status = jira_data.get("status", "unknown")
        emoji = {"connected": "🟢", "unconfigured": "⚪", "failed": "🔴"}.get(status, "⚪")
        st.sidebar.info(f"{emoji} {jira_data.get('message', status)}")
    else:
        st.sidebar.error(f"检查失败：{jira_data}")

st.sidebar.divider()
st.sidebar.caption("v1.0.0 | AI-Driven R&D Agent")

# ==================== 页面路由 ====================
pages = [
    st.Page("pages/knowledge.py", title="知识库管理", icon="📚"),
    st.Page("pages/chat.py", title="智能问答", icon="💬"),
    st.Page("pages/analysis.py", title="日志分析", icon="📊"),
    st.Page("pages/test_cases.py", title="测试用例生成", icon="📝"),
    st.Page("pages/testing.py", title="自动化测试", icon="🧪"),
    st.Page("pages/agent.py", title="Agent 执行", icon="🤖"),
    st.Page("pages/history.py", title="历史报告", icon="📋"),
]

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
