"""自动化测试页。"""
from __future__ import annotations

import base64
import time

import streamlit as st

from utils.api import api_get, api_post, get_api_host


def _clear_test_task_state():
    """清理测试任务相关 session_state。"""
    st.session_state.pop("test_task_id", None)
    st.session_state.pop("test_task_start", None)
    st.session_state.pop("test_task_cancelling", None)


def render() -> None:
    API_BASE = get_api_host()

    st.title("🧪 自动化测试")
    st.markdown("基于 Selenium 的自动化测试执行。选择场景并启动测试，查看实时结果和 AI 分析。")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("测试配置")
        # 默认使用后端内置的 Demo 测试站点（含登录/搜索/下单表单）
        _demo_url = f"{API_BASE}/demo"
        if "test_base_url" not in st.session_state or not st.session_state.test_base_url:
            st.session_state.test_base_url = _demo_url
        base_url = st.text_input(
            "目标网站",
            value=st.session_state.test_base_url,
            help="默认使用内置 Demo 站点（/demo）。也可改为任意真实网站地址。",
        )
        st.session_state.test_base_url = base_url

        # 浏览器模式选择：无头 vs 可见窗口
        browser_mode = st.radio(
            "浏览器模式",
            options=["headless", "visible"],
            format_func=lambda x: {
                "headless": "🤫 无头模式（后台运行，速度快）",
                "visible": "🖥️ 可见模式（弹出浏览器窗口，可观察测试过程）",
            }[x],
            help="无头模式：Chrome 在后台运行，不显示窗口，适合 CI/CD\n可见模式：会弹出真实 Chrome 窗口，可以看到测试操作的整个过程",
        )
        headless = browser_mode == "headless"
        timeout = st.slider("超时时间（秒）", 10, 60, 30)

        # ---- Selenium 环境诊断（手动触发，避免每次 rerun 都调 Chrome 子进程） ----
        st.subheader("环境诊断")
        if st.button("🔍 检测 Chrome / chromedriver", key="selenium_diag_btn", help="手动触发一次环境检测"):
            with st.spinner("检测中..."):
                success, data = api_get("system/selenium-diagnose", timeout=10)
                if success:
                    st.session_state.selenium_diag_result = data
                else:
                    st.session_state.selenium_diag_result = {"error": str(data)[:200]}
                st.rerun()
        if diag := st.session_state.get("selenium_diag_result"):
            if diag.get("ready"):
                st.success(f"✅ 环境就绪 | Chrome v{diag.get('chrome_version')} | chromedriver v{diag.get('chromedriver_version') or '自动管理'}")
                for tip in diag.get("tips", []):
                    st.info(f"💡 {tip}")
            elif diag.get("error"):
                st.warning(f"诊断失败: {diag['error']}")
                st.info('提示：开启下方"沙盒模式"可不依赖 Chrome 直接体验测试流程')
            else:
                st.warning(f"⚠️ {diag.get('message', '环境未就绪')}")
                with st.expander("查看诊断详情"):
                    st.json(diag)
                st.info('提示：开启下方"沙盒模式"可不依赖 Chrome 直接体验测试流程')

        st.subheader("选择测试场景")

        # 测试模式选择：预定义场景 vs 自定义场景
        test_mode = st.radio(
            "测试模式",
            options=["preset", "custom"],
            format_func=lambda x: {"preset": "📋 预定义场景（内置 Demo）", "custom": "🔧 自定义场景（任意网站）"}[x],
            help="预定义场景：测试内置 Demo 站点\n自定义场景：可测任意网站（如百度、搜狗等）",
        )

        if test_mode == "preset":
            scenarios = st.multiselect(
                "要执行的测试",
                options=["login", "search", "order"],
                default=["login", "search"],
                format_func=lambda x: {"login": "🔑 登录流程", "search": "🔍 搜索流程", "order": "🛒 下单流程"}[x],
            )
            custom_scenarios_json = None
        else:
            # ---- 自定义场景编辑器 ----
            st.markdown("##### 自定义测试步骤")
            st.caption("按顺序添加测试步骤，每步选择操作类型和定位方式")

            # 初始化步骤列表
            if "custom_steps" not in st.session_state:
                st.session_state.custom_steps = [
                    {"action": "navigate", "by": "id", "value": "", "description": ""},
                ]

            # 预设模板按钮：缩短标签、增加间距，避免在窄栏里换行拥挤
            tpl_cols = st.columns([1, 1, 1])
            with tpl_cols[0]:
                if st.button("🔎 百度模板", use_container_width=True, key="tpl_baidu", help="加载百度搜索测试步骤"):
                    st.session_state.custom_steps = [
                        {"action": "navigate", "by": "id", "value": "https://www.baidu.com", "description": "打开百度首页"},
                        {"action": "input", "by": "id", "value": "kw::Selenium自动化测试", "description": "输入搜索关键词"},
                        {"action": "click", "by": "id", "value": "su", "description": "点击搜索按钮"},
                        {"action": "wait", "by": "id", "value": "2", "description": "等待页面加载"},
                        {"action": "verify", "by": "css_selector", "value": "#content_left", "description": "验证搜索结果"},
                    ]
                    st.rerun()
            with tpl_cols[1]:
                if st.button("🔍 搜狗模板", use_container_width=True, key="tpl_sogou", help="加载搜狗搜索测试步骤"):
                    st.session_state.custom_steps = [
                        {"action": "navigate", "by": "id", "value": "https://www.sogou.com", "description": "打开搜狗首页"},
                        {"action": "input", "by": "id", "value": "query::Selenium自动化测试", "description": "输入搜索关键词"},
                        {"action": "click", "by": "id", "value": "stb", "description": "点击搜索按钮"},
                        {"action": "wait", "by": "id", "value": "2", "description": "等待页面加载"},
                        {"action": "verify", "by": "css_selector", "value": ".results", "description": "验证搜索结果"},
                    ]
                    st.rerun()
            with tpl_cols[2]:
                if st.button("🗑️ 清空", use_container_width=True, key="tpl_clear", help="清空所有自定义步骤"):
                    st.session_state.custom_steps = [{"action": "navigate", "by": "id", "value": "", "description": ""}]
                    st.rerun()

            st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)

            # 表头（更宽松的列宽，避免在窄栏里换行/挤压）
            hdr = st.columns([1.2, 1.2, 3.0, 3.0, 0.6])
            hdr[0].markdown('<p class="step-header">操作</p>', unsafe_allow_html=True)
            hdr[1].markdown('<p class="step-header">定位方式</p>', unsafe_allow_html=True)
            hdr[2].markdown('<p class="step-header">值 / 输入</p>', unsafe_allow_html=True)
            hdr[3].markdown('<p class="step-header">步骤描述</p>', unsafe_allow_html=True)
            hdr[4].markdown('<p class="step-header">删除</p>', unsafe_allow_html=True)

            action_options = ["navigate", "input", "click", "verify", "wait"]
            action_labels = {"navigate": "🌐 打开URL", "input": "⌨️ 输入文本", "click": "🖱️ 点击", "verify": "✅ 验证元素", "wait": "⏳ 等待"}
            by_options = ["id", "name", "xpath", "css_selector", "class_name", "tag_name", "link_text"]
            placeholder_map = {
                "navigate": "URL，如 https://www.baidu.com",
                "input": "定位值::输入内容，如 kw::搜索词",
                "click": "定位值，如 su",
                "verify": "定位值，如 #content_left",
                "wait": "等待秒数，如 2",
            }

            new_steps = []
            for i, step in enumerate(st.session_state.custom_steps):
                # 每个步骤使用带边框的容器，视觉上更清晰，避免挤在一起
                with st.container(border=True):
                    row_cols = st.columns([1.2, 1.2, 3.0, 3.0, 0.6])

                    with row_cols[0]:
                        action = st.selectbox(
                            "操作", action_options,
                            index=action_options.index(step.get("action", "navigate")),
                            format_func=lambda x: action_labels[x],
                            key=f"step_action_{i}",
                            label_visibility="collapsed",
                        )
                    with row_cols[1]:
                        if action in ("navigate", "wait"):
                            by_val = st.selectbox("定位", ["—"], key=f"step_by_{i}", disabled=True, label_visibility="collapsed")
                        else:
                            by_val = st.selectbox(
                                "定位", by_options,
                                index=by_options.index(step.get("by", "id")),
                                key=f"step_by_{i}",
                                label_visibility="collapsed",
                            )
                    with row_cols[2]:
                        val = st.text_input(
                            "值", value=step.get("value", ""),
                            placeholder=placeholder_map.get(action, ""),
                            key=f"step_value_{i}",
                            label_visibility="collapsed",
                        )
                    with row_cols[3]:
                        desc = st.text_input(
                            "描述", value=step.get("description", ""),
                            placeholder=f"步骤 {i+1} 描述",
                            key=f"step_desc_{i}",
                            label_visibility="collapsed",
                        )
                    with row_cols[4]:
                        if len(st.session_state.custom_steps) > 1:
                            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
                            if st.button("✕", key=f"step_del_{i}", help="删除此步骤", type="tertiary"):
                                st.session_state.custom_steps.pop(i)
                                st.rerun()
                        else:
                            st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)

                new_steps.append({
                    "action": action,
                    "by": by_val if by_val != "—" else "id",
                    "value": val,
                    "description": desc,
                })

            add_col, count_col = st.columns([1, 4])
            with add_col:
                if st.button("➕ 添加步骤", use_container_width=True, key="add_step"):
                    st.session_state.custom_steps.append({"action": "navigate", "by": "id", "value": "", "description": ""})
                    st.rerun()
            with count_col:
                if new_steps:
                    st.caption(f"共 {len(new_steps)} 个步骤")

            st.session_state.custom_steps = new_steps
            scenarios = []
            custom_scenarios_json = new_steps

            # 显示步骤预览
            if new_steps:
                with st.expander("查看步骤 JSON"):
                    st.json(new_steps)

        auto_analyze = st.checkbox("失败时自动AI分析", value=True)
        sandbox = st.checkbox(
            "🧪 沙盒模式（无需Chrome，模拟执行）",
            value=True,
            help="开启后使用模拟的 Selenium 执行（推荐）。关闭则驱动真实 Chrome 浏览器。",
        )

        # 按钮禁用条件：预定义模式需要选场景，自定义模式需要有步骤；任务执行中禁用
        task_running = bool(st.session_state.get("test_task_id"))
        btn_disabled = (
            (test_mode == "preset" and not scenarios)
            or (test_mode == "custom" and not custom_scenarios_json)
            or task_running
        )

        # ---- 异步测试执行 ----
        if st.button("▶ 开始测试", type="primary", disabled=btn_disabled, key="test_start_btn"):
            payload = {
                "scenarios": scenarios if test_mode == "preset" else ["custom"],
                "base_url": base_url,
                "headless": headless,
                "timeout_seconds": timeout,
                "auto_analyze": auto_analyze,
                "sandbox": sandbox,
            }
            if test_mode == "custom" and custom_scenarios_json:
                payload["custom_scenarios"] = [{
                    "name": "自定义场景",
                    "steps": custom_scenarios_json,
                }]

            success, data = api_post(
                "api/v1/testing/run/async",
                json=payload,
                timeout=30,
            )
            if success:
                st.session_state.test_task_id = data["task_id"]
                st.session_state.test_task_start = time.time()
                st.rerun()
            else:
                st.error(f"创建测试任务失败: {data}")

        # 轮询异步任务状态
        if st.session_state.get("test_task_id"):
            task_id = st.session_state.test_task_id
            elapsed = int(time.time() - st.session_state.get("test_task_start", time.time()))
            mode_label = "沙盒模式" if sandbox else ("无头浏览器" if headless else "可见浏览器")

            # 显示取消按钮
            cancel_col, _ = st.columns([1, 2])
            with cancel_col:
                if st.button("⏹ 取消测试", key="test_cancel_btn", type="secondary"):
                    success, cancel_data = api_post(
                        f"api/v1/testing/tasks/{task_id}/cancel",
                        timeout=10,
                    )
                    if success:
                        st.info("已发送取消请求，等待任务停止...")
                        st.session_state.test_task_cancelling = True
                        st.rerun()
                    else:
                        st.warning(f"取消请求失败: {cancel_data}")

            with st.spinner(f"测试执行中... ({mode_label}) 已耗时 {elapsed}s"):
                success, task = api_get(
                    f"api/v1/testing/tasks/{task_id}",
                    timeout=10,
                )
                if success:
                    if task["status"] == "completed":
                        st.session_state.test_report = task["result"]
                        _clear_test_task_state()
                        st.rerun()
                    elif task["status"] == "cancelled":
                        st.session_state.test_report = task["result"]
                        st.warning("测试任务已取消")
                        _clear_test_task_state()
                        st.rerun()
                    elif task["status"] == "failed":
                        st.error(f"测试执行失败: {task.get('error', '未知错误')}")
                        _clear_test_task_state()
                    else:
                        time.sleep(2)
                        st.rerun()
                else:
                    st.error(f"查询任务状态失败: {task}")
                    _clear_test_task_state()

    with col2:
        st.subheader("测试结果")
        if "test_report" in st.session_state and st.session_state.test_report:
            report = st.session_state.test_report

            # 统计卡片
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("总场景", report.get("total_scenarios", 0))
            with m2:
                st.metric("✅ 通过", report.get("passed_count", 0))
            with m3:
                st.metric("❌ 失败", report.get("failed_count", 0))
            with m4:
                rate = report.get("pass_rate", 0)
                st.metric("通过率", f"{rate:.0%}")

            # 沙盒模式标识
            if sandbox:
                st.info("🧪 当前为沙盒模式 — 测试结果由 Mock 引擎生成，用于演示流程")
            else:
                st.warning("🌐 真实浏览器模式 — 需要 Chrome 和 chromedriver")

            # 各用例详情
            for result in report.get("results", []):
                status = result.get("status", "unknown")
                icon = {"passed": "✅", "failed": "❌", "error": "⚠️", "skipped": "⏭️"}.get(status, "⚪")
                with st.expander(
                    f"{icon} {result.get('scenario', 'unknown')} — "
                    f"{result.get('duration_ms', 0):.0f}ms"
                ):
                    for step in result.get("steps", []):
                        step_status = step.get("status", "unknown")
                        step_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}.get(step_status, "⚪")
                        st.markdown(
                            f"{step_icon} **{step.get('step_name', '')}** "
                            f"({step.get('duration_ms', 0):.0f}ms)"
                        )
                        if step.get("error_message"):
                            st.error(step["error_message"])

                    # 显示 Selenium 日志
                    if result.get("selenium_logs"):
                        with st.expander("📋 Selenium 日志"):
                            st.code(result["selenium_logs"])

                    # 显示失败截图
                    screenshot = result.get("screenshot_base64", "")
                    if screenshot and status in ("failed", "error"):
                        try:
                            img_bytes = base64.b64decode(screenshot)
                            with st.expander("📸 失败截图"):
                                st.image(
                                    img_bytes,
                                    caption="失败时的页面截图",
                                    use_container_width=True,
                                )
                        except Exception as e:
                            st.warning(f"截图解析失败: {e}")

            # 失败分析
            if report.get("failure_analysis"):
                st.subheader("🤖 AI 失败分析")
                for analysis in report["failure_analysis"]:
                    st.warning(analysis)
        else:
            st.info("👈 请在左侧配置并启动测试")


if __name__ == "__main__":
    render()
