"""日志分析页。"""
from __future__ import annotations

import streamlit as st

from utils.api import api_get, api_post, get_api_host


def render() -> None:
    API_BASE = get_api_host()

    st.title("📊 日志分析")
    st.markdown("上传测试日志文件、粘贴日志内容，或直接读取后端运行日志，AI 自动识别异常并生成分析报告。")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("输入日志")

        # 快捷入口：一键分析最近错误
        quick_cols = st.columns(2)
        if quick_cols[0].button("🚨 刚才什么错误？", key="quick_recent_error"):
            st.session_state.input_method_radio = "📋 读取运行日志"
            st.session_state.input_method = "📋 读取运行日志"
            st.session_state.rt_level = "ERROR"
            st.session_state.rt_auto_level = "ERROR"
            st.session_state.rt_auto_fetch = True
            st.rerun()
        if quick_cols[1].button("🧹 清空日志", key="quick_clear_log"):
            st.session_state.rt_log_content = ""
            st.session_state.rt_log_filename = "runtime_app.log"
            st.session_state.analysis_result = {}
            st.rerun()

        input_method = st.radio(
            "输入方式",
            ["📄 上传文件", "✏️ 粘贴内容", "📋 读取运行日志"],
            index=["📄 上传文件", "✏️ 粘贴内容", "📋 读取运行日志"].index(
                st.session_state.get("input_method", "📋 读取运行日志")
            ),
            key="input_method_radio",
        )
        # 同步到 session_state，方便快捷按钮控制
        st.session_state.input_method = input_method

        # 从 session_state 恢复日志内容（解决 Streamlit rerun 导致局部变量丢失的问题）
        if "rt_log_content" not in st.session_state:
            st.session_state.rt_log_content = ""
            st.session_state.rt_log_filename = "runtime_app.log"
            st.session_state.rt_auto_fetched = False

        log_content = ""
        filename = "manual_input.log"

        if input_method == "📄 上传文件":
            # 切换到其他模式时清空运行日志缓存
            if st.session_state.rt_log_content:
                st.session_state.rt_log_content = ""
            log_file = st.file_uploader(
                "选择日志文件",
                type=["log", "txt"],
                help="支持 .log 和 .txt 格式",
            )
            if log_file:
                raw_bytes = log_file.getvalue()
                try:
                    log_content = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    log_content = raw_bytes.decode("gbk", errors="replace")
                filename = log_file.name
                st.text_area("日志预览", value=log_content[:3000], height=250, disabled=True)
                st.caption(f"文件大小: {len(log_content)} 字符")

        elif input_method == "✏️ 粘贴内容":
            # 切换到其他模式时清空运行日志缓存
            if st.session_state.rt_log_content:
                st.session_state.rt_log_content = ""
            log_content = st.text_area(
                "粘贴日志内容",
                height=250,
                placeholder="在此粘贴测试日志...\n\n例如：\nTraceback (most recent call last):\n  File 'test.py', line 10\nTimeoutException: Page load timeout",
            )

        else:  # 读取运行日志
            st.caption("直接拉取后端服务 `data/logs/app.log` 的最近日志，无需上传文件。")
            rt_col1, rt_col2 = st.columns(2)
            with rt_col1:
                rt_tail = st.slider("读取行数", 50, 2000, 200, step=50, key="rt_tail")
            with rt_col2:
                # 快捷按钮可能设置了默认 ERROR 级别
                _default_level = st.session_state.get("rt_auto_level", "all")
                _level_options = ["all", "ERROR", "WARNING", "INFO"]
                _level_index = _level_options.index(_default_level) if _default_level in _level_options else 0
                rt_level = st.selectbox(
                    "日志级别",
                    options=_level_options,
                    index=_level_index,
                    key="rt_level",
                    help="ERROR 只看错误；all 看全部。问'刚才什么错误'建议选 ERROR",
                )

            # 自动拉取：首次进入页面且没有日志内容时，自动拉取最近 ERROR 日志
            _should_auto_fetch = (
                not st.session_state.get("rt_auto_fetched", False)
                and not st.session_state.rt_log_content.strip()
            ) or st.session_state.get("rt_auto_fetch", False)

            if _should_auto_fetch:
                st.session_state.rt_auto_fetch = False
                st.session_state.rt_auto_fetched = True
                with st.spinner("正在自动读取运行日志..."):
                    success, data = api_get(
                        "api/v1/analysis/runtime-logs",
                        params={"tail_lines": rt_tail, "level": rt_level},
                        timeout=30,
                    )
                    if success:
                        st.session_state.rt_log_content = data.get("content", "")
                        st.session_state.rt_log_filename = "runtime_app.log"
                        st.toast(
                            f"已自动读取 {data.get('returned_lines', 0)} 行 "
                            f"{data.get('level', 'all')} 日志",
                            icon="📋",
                        )
                    else:
                        st.error(f"自动读取失败: {data}")

            if st.button("📥 拉取运行日志", key="fetch_rt_log"):
                with st.spinner("正在读取运行日志..."):
                    success, data = api_get(
                        "api/v1/analysis/runtime-logs",
                        params={"tail_lines": rt_tail, "level": rt_level},
                        timeout=30,
                    )
                    if success:
                        # 持久化到 session_state，避免 rerun 后丢失
                        st.session_state.rt_log_content = data.get("content", "")
                        st.session_state.rt_log_filename = "runtime_app.log"
                        st.success(
                            f"已读取 {data.get('returned_lines', 0)} 行 "
                            f"(共 {data.get('total_lines', 0)} 行，过滤级别: {data.get('level', 'all')})"
                        )
                    else:
                        st.error(f"读取失败: {data}")

            # 从 session_state 恢复日志内容
            log_content = st.session_state.rt_log_content
            filename = st.session_state.rt_log_filename

            if log_content:
                st.text_area("运行日志预览", value=log_content[:3000], height=250, disabled=True)
                st.caption(f"已加载: {len(log_content)} 字符")
                # 提供清空按钮
                if st.button("🗑️ 清空已拉取的日志", key="clear_rt_log"):
                    st.session_state.rt_log_content = ""
                    st.session_state.rt_log_filename = "runtime_app.log"
                    st.rerun()

        # 分析参数
        include_historical = st.checkbox("检索历史相似案例", value=True)
        top_k = st.slider("检索案例数", 1, 10, 3, key="analysis_topk")

        if st.button("🔍 开始分析", type="primary", disabled=not log_content.strip()):
            with st.spinner("正在分析日志..."):
                success, result = api_post(
                    "api/v1/analysis/log",
                    data={
                        "log_content": log_content,
                        "filename": filename,
                        "include_historical": include_historical,
                        "top_k": top_k,
                    },
                    timeout=180,
                )
                if success:
                    st.session_state.analysis_result = result
                    st.rerun()
                else:
                    st.error(f"分析失败: {result}")

    with col2:
        st.subheader("分析结果")
        if "analysis_result" in st.session_state and st.session_state.analysis_result:
            result = st.session_state.analysis_result

            # 严重等级
            severity = result.get("severity", "中")
            severity_color = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(severity, "⚪")
            st.markdown(f"### 严重等级: {severity_color} {severity}")

            # 问题摘要
            st.markdown("### 📋 问题摘要")
            st.info(result.get("summary", "无"))

            # 检测到的异常
            exceptions = result.get("exceptions_found", [])
            if exceptions:
                st.markdown("### 🐛 检测到的异常")
                for exc in exceptions:
                    with st.expander(f"{exc.get('exception_type', 'Unknown')}"):
                        st.markdown(f"**消息**: {exc.get('message', 'N/A')}")
                        if exc.get("file_path"):
                            st.markdown(f"**位置**: {exc['file_path']}:{exc.get('line_number', '?')}")
                        traceback_lines = exc.get("traceback_lines", [])
                        if traceback_lines:
                            st.code("\n".join(traceback_lines), language="python")

            # 可能原因
            causes = result.get("possible_causes", [])
            if causes:
                st.markdown("### 🔎 可能原因")
                for cause in causes:
                    st.markdown(f"- {cause}")

            # 历史案例
            hist_cases = result.get("historical_cases", [])
            if hist_cases:
                st.markdown("### 📚 历史相似案例")
                for case in hist_cases:
                    with st.expander(
                        f"{case.get('title', '案例')[:50]}... (相似度: {case.get('similarity_score', 0):.2f})"
                    ):
                        st.markdown(case.get("description", "")[:500])

            # 修复建议
            fixes = result.get("fix_suggestions", [])
            if fixes:
                st.markdown("### ✅ 修复建议")
                for fix in fixes:
                    st.markdown(f"- {fix}")

            # 原始分析
            raw = result.get("raw_analysis", "")
            if raw:
                with st.expander("📝 查看完整分析"):
                    st.markdown(raw)

            # ---- JIRA 缺陷创建联动 ----
            st.divider()
            st.subheader("🎫 创建 JIRA 缺陷单")
            jira_col1, jira_col2 = st.columns(2)
            with jira_col1:
                jira_title = st.text_input(
                    "缺陷标题",
                    value=f"[AI分析] {result.get('summary', '')[:80]}",
                    key="jira_title",
                )
                jira_priority = st.selectbox(
                    "优先级",
                    ["Highest", "High", "Medium", "Low"],
                    index={"高": 1, "中": 2, "低": 3}.get(result.get("severity", "中"), 2),
                    key="jira_priority",
                )
            with jira_col2:
                jira_assignee = st.text_input("指派人（可选）", key="jira_assignee")
                jira_labels = st.text_input("标签（逗号分隔）", value="ai-generated,bug,automation", key="jira_labels")

            if st.button("🚀 创建 JIRA 缺陷单", type="primary", key="create_jira_btn"):
                with st.spinner("正在创建 JIRA 缺陷单..."):
                    # 构建缺陷描述
                    causes_text = "\n".join(f"- {c}" for c in result.get("possible_causes", []))
                    fixes_text = "\n".join(f"- {f}" for f in result.get("fix_suggestions", []))
                    description = f"""## AI 分析结果

### 问题摘要
{result.get('summary', '')}

### 检测到的异常
{chr(10).join(f"- {e.get('exception_type', '')}: {e.get('message', '')[:200]}" for e in result.get('exceptions_found', []))}

### 可能原因
{causes_text}

### 修复建议
{fixes_text}

### 严重等级
{result.get('severity', '')}
"""
                    success, jira_data = api_post(
                        "api/v1/jira/create",
                        json={
                            "title": jira_title,
                            "description": description,
                            "priority": jira_priority,
                            "log_content": log_content[:3000],
                            "ai_analysis": result.get("raw_analysis", "")[:3000],
                            "assignee": jira_assignee,
                            "labels": [l.strip() for l in jira_labels.split(",") if l.strip()],
                        },
                        timeout=120,
                    )
                    if success:
                        if jira_data.get("status") == "success":
                            st.success(f"✅ 缺陷单已创建: {jira_data.get('issue_key', '')}")
                            if jira_data.get("issue_url"):
                                st.info(f"🔗 {jira_data['issue_url']}")
                        else:
                            st.warning(f"⚠️ {jira_data.get('message', '创建结果未知')}")
                    else:
                        st.error(f"创建失败: {jira_data}")
        else:
            st.info("👈 请在左侧提交日志进行分析")


if __name__ == "__main__":
    render()
