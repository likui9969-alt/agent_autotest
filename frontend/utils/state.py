"""Streamlit session_state 读写 helper。"""
from __future__ import annotations

from typing import Any

import streamlit as st


def get_state(key: str, default: Any = None) -> Any:
    """安全读取 session_state 中的 key。"""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """写入 session_state。"""
    st.session_state[key] = value


def init_state() -> None:
    """初始化所有页面依赖的 session_state 默认值。"""
    from utils.api import _DEFAULT_API_BASE, auto_detect_backend

    defaults = {
        "api_host": _DEFAULT_API_BASE,
        "messages": [],
        "input_method": "📋 读取运行日志",
        "rt_log_content": "",
        "rt_log_filename": "runtime_app.log",
        "rt_auto_fetched": False,
        "analysis_result": {},
        "test_base_url": None,  # 页面内根据 api_host 计算
        "custom_steps": [
            {"action": "navigate", "by": "id", "value": "", "description": ""},
        ],
        "selenium_diag": None,
        "test_report": None,
        "agent_task": "",
        "agent_events": [],
        "agent_result": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 如果当前地址仍为默认值，尝试自动探测一次（减少用户手动配置）
    if st.session_state.api_host == _DEFAULT_API_BASE:
        detected = auto_detect_backend()
        if detected != _DEFAULT_API_BASE:
            st.session_state.api_host = detected
