"""统一后端 API 调用封装。

提供基于当前 st.session_state.api_host 的 get/post  helper，统一处理
ConnectionError 并返回 (success, data_or_error) 元组，方便页面直接渲染提示。
"""
from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


_DEFAULT_API_BASE = os.environ.get("AGENT_API_BASE", "http://localhost:8000")


class ApiError(Exception):
    """后端 API 调用异常。"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def get_api_host() -> str:
    """返回当前生效的后端地址（去尾斜杠）。"""
    return str(st.session_state.get("api_host", _DEFAULT_API_BASE)).rstrip("/")


def _build_url(path: str) -> str:
    host = get_api_host()
    if path.startswith("/"):
        path = path[1:]
    return f"{host}/{path}"


def _handle_connection_error(host: str) -> tuple[bool, str]:
    return False, f"❌ 无法连接到后端 {host}，请确认服务已启动"


def _normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """确保默认超时存在，避免请求无限挂起。"""
    if "timeout" not in kwargs:
        kwargs["timeout"] = 30
    return kwargs


def api_get(path: str, **kwargs) -> tuple[bool, Any]:
    """向后端发送 GET 请求。

    Returns:
        (True, 解析后的 JSON 数据) 或 (False, 错误信息字符串)
    """
    url = _build_url(path)
    kwargs = _normalize_kwargs(kwargs)
    try:
        resp = requests.get(url, **kwargs)
    except requests.ConnectionError:
        return _handle_connection_error(get_api_host())
    except Exception as e:
        return False, f"请求异常: {e}"

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    if resp.status_code == 200:
        return True, data
    return False, _extract_error(data, resp.status_code)


def api_post(path: str, **kwargs) -> tuple[bool, Any]:
    """向后端发送 POST 请求。

    Returns:
        (True, 解析后的 JSON 数据) 或 (False, 错误信息字符串)
    """
    url = _build_url(path)
    kwargs = _normalize_kwargs(kwargs)
    try:
        resp = requests.post(url, **kwargs)
    except requests.ConnectionError:
        return _handle_connection_error(get_api_host())
    except Exception as e:
        return False, f"请求异常: {e}"

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    if resp.status_code == 200:
        return True, data
    return False, _extract_error(data, resp.status_code)


def api_post_raw(path: str, **kwargs) -> requests.Response:
    """向后端发送 POST 请求并返回原始 Response（用于 SSE 等场景）。

    调用方需自行处理异常。
    """
    url = _build_url(path)
    kwargs = _normalize_kwargs(kwargs)
    return requests.post(url, **kwargs)


def _extract_error(data: Any, status_code: int) -> str:
    if isinstance(data, dict):
        msg = data.get("message") or data.get("detail") or data.get("error")
        if msg:
            return f"请求失败 ({status_code}): {str(msg)[:300]}"
    text = str(data)[:300]
    return f"请求失败 ({status_code}): {text}"


def probe_backend(host: str | None = None, timeout: float = 1.0) -> bool:
    """轻量探测后端健康接口（/health/lite，不调 LLM/Chrome）。"""
    host = (host or get_api_host()).rstrip("/")
    try:
        r = requests.get(f"{host}/health/lite", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def auto_detect_backend() -> str:
    """自动扫描本机 8000-8005 端口，返回第一个在线的后端地址。"""
    for port in range(8000, 8006):
        url = f"http://localhost:{port}"
        if probe_backend(url):
            return url
    return _DEFAULT_API_BASE
