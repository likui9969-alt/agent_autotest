"""
Streamlit 前端应用 — AI-Driven 研发效能智能体
提供 4 个功能页面：知识库管理、智能问答、日志分析、自动化测试
启动命令：streamlit run app.py
"""
import os
import streamlit as st
import requests
import json

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AI研发效能智能体",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 后端 API 地址默认值：环境变量 AGENT_API_BASE > 8000
_DEFAULT_API_BASE = os.environ.get("AGENT_API_BASE", "http://localhost:8000")


def _probe_backend(host: str, timeout: float = 1.5) -> bool:
    """探测指定地址的后端是否在线（调用 /health 接口）"""
    try:
        r = requests.get(f"{host}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _auto_detect_backend() -> str:
    """自动扫描本机 8000-8005 端口，返回第一个在线的后端地址"""
    for port in range(8000, 8006):
        url = f"http://localhost:{port}"
        if _probe_backend(url):
            return url
    return _DEFAULT_API_BASE


# ==================== 侧边栏导航 ====================
st.sidebar.title("🤖 AI研发效能智能体")
st.sidebar.markdown("基于 RAG 的自动化测试与故障分析系统")
st.sidebar.divider()

page = st.sidebar.radio(
    "导航菜单",
    ["📚 知识库管理", "💬 智能问答", "📊 日志分析", "🧪 自动化测试", "🤖 Agent 执行"],
)

# API 地址配置 — 持久化到 session_state，避免重跑时被重置
if "api_host" not in st.session_state:
    st.session_state.api_host = _DEFAULT_API_BASE

# 自动探测按钮 + 手动输入
_detect_col, _status_col = st.sidebar.columns([1, 1])
if _detect_col.button("🔍 自动探测", key="detect_btn", help="扫描 8000-8005 端口自动找到后端"):
    with st.spinner("扫描中..."):
        detected = _auto_detect_backend()
        st.session_state.api_host = detected
        st.rerun()

# 实时连接状态指示灯
if _probe_backend(st.session_state.api_host, timeout=1.0):
    _status_col.markdown("🟢 **已连接**")
else:
    _status_col.markdown("🔴 **未连接**")

api_host = st.sidebar.text_input("后端API地址", value=st.session_state.api_host, key="api_host_input")
if api_host:
    st.session_state.api_host = api_host.rstrip("/")
API_BASE = st.session_state.api_host

st.sidebar.divider()
st.sidebar.caption("v1.0.0 | AI-Driven R&D Agent")

# ==================== 页面1：知识库管理 ====================
if page == "📚 知识库管理":
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
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        resp = requests.post(
                            f"{API_BASE}/api/v1/knowledge/upload",
                            files=files,
                            timeout=120,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(f"✅ 文档 '{data['filename']}' 上传成功！")
                            st.info(f"切分为 {data['chunk_count']} 个文本块")
                            st.rerun()
                        else:
                            st.error(f"上传失败: {resp.json().get('message', resp.text)}")
                    except requests.ConnectionError:
                        st.error(f"❌ 无法连接到后端 {API_BASE}，请确认服务已启动")
                    except Exception as e:
                        st.error(f"上传异常: {str(e)}")

        # ---- 重建向量库 ----
        st.subheader("重建向量库")
        st.caption("清空所有已有向量，重新索引 data/docs/ 目录下的所有文档")
        if st.button("🔄 重建向量库", type="secondary"):
            with st.spinner("正在重建向量库..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/v1/knowledge/rebuild",
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(data.get("message", "重建完成"))
                        st.rerun()
                    else:
                        st.error(f"重建失败: {resp.text}")
                except requests.ConnectionError:
                    st.error(f"❌ 无法连接到后端 {API_BASE}")

    with col2:
        # ---- 知识库统计 ----
        st.subheader("知识库概况")
        if st.button("🔄 刷新统计"):
            st.rerun()

        try:
            resp = requests.get(f"{API_BASE}/api/v1/knowledge/stats", timeout=10)
            if resp.status_code == 200:
                stats = resp.json()
                st.metric("文档总数", stats.get("total_documents", 0))
                st.metric("向量块数", stats.get("total_chunks", 0))
                st.metric("集合名称", stats.get("collection_name", "N/A"))
            else:
                st.warning("无法获取统计信息")
        except requests.ConnectionError:
            st.warning(f"⚠ 后端服务未连接\n\n请先启动后端：\n```\nuvicorn backend.main:app --reload\n```")

# ==================== 页面2：智能问答 ====================
elif page == "💬 智能问答":
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
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/v1/rag/query",
                        json={
                            "question": question,
                            "top_k": top_k,
                            "search_type": search_type,
                            "include_sources": include_sources,
                        },
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
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
                        error_text = resp.text[:200]
                        st.error(f"查询失败: {error_text}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ 查询失败: {error_text}",
                            "sources": [],
                        })
                except requests.ConnectionError:
                    st.error(f"❌ 无法连接到后端 {API_BASE}，请确认服务已启动")
                except Exception as e:
                    st.error(f"查询异常: {str(e)}")

    # 清空对话按钮
    if st.session_state.messages and st.button("🗑 清空对话", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

# ==================== 页面3：日志分析 ====================
elif page == "📊 日志分析":
    st.title("📊 日志分析")
    st.markdown("上传测试日志文件、粘贴日志内容，或直接读取后端运行日志，AI 自动识别异常并生成分析报告。")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("输入日志")
        input_method = st.radio("输入方式", ["📄 上传文件", "✏️ 粘贴内容", "📋 读取运行日志"])

        # 从 session_state 恢复日志内容（解决 Streamlit rerun 导致局部变量丢失的问题）
        if "rt_log_content" not in st.session_state:
            st.session_state.rt_log_content = ""
            st.session_state.rt_log_filename = "runtime_app.log"

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
                rt_level = st.selectbox(
                    "日志级别",
                    options=["all", "ERROR", "WARNING", "INFO"],
                    index=0,
                    key="rt_level",
                    help="ERROR 只看错误；all 看全部。问'刚才什么错误'建议选 ERROR",
                )

            if st.button("📥 拉取运行日志", key="fetch_rt_log"):
                with st.spinner("正在读取运行日志..."):
                    try:
                        resp = requests.get(
                            f"{API_BASE}/api/v1/analysis/runtime-logs",
                            params={"tail_lines": rt_tail, "level": rt_level},
                            timeout=30,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            # 持久化到 session_state，避免 rerun 后丢失
                            st.session_state.rt_log_content = data.get("content", "")
                            st.session_state.rt_log_filename = "runtime_app.log"
                            st.success(
                                f"已读取 {data.get('returned_lines', 0)} 行 "
                                f"(共 {data.get('total_lines', 0)} 行，过滤级别: {data.get('level', 'all')})"
                            )
                        else:
                            st.error(f"读取失败: {resp.text[:300]}")
                    except requests.ConnectionError:
                        st.error(f"❌ 无法连接到后端 {API_BASE}")
                    except Exception as e:
                        st.error(f"读取异常: {str(e)}")

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
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/v1/analysis/log",
                        data={
                            "log_content": log_content,
                            "filename": filename,
                            "include_historical": include_historical,
                            "top_k": top_k,
                        },
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state.analysis_result = result
                        st.rerun()
                    else:
                        st.error(f"分析失败: {resp.text[:300]}")
                except requests.ConnectionError:
                    st.error(f"❌ 无法连接到后端 {API_BASE}")
                except Exception as e:
                    st.error(f"分析异常: {str(e)}")

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
                    try:
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
                        jira_resp = requests.post(
                            f"{API_BASE}/api/v1/jira/create",
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
                        if jira_resp.status_code == 200:
                            jira_data = jira_resp.json()
                            if jira_data.get("status") == "success":
                                st.success(f"✅ 缺陷单已创建: {jira_data.get('issue_key', '')}")
                                if jira_data.get("issue_url"):
                                    st.info(f"🔗 {jira_data['issue_url']}")
                            else:
                                st.warning(f"⚠️ {jira_data.get('message', '创建结果未知')}")
                        else:
                            st.error(f"创建失败: {jira_resp.text[:300]}")
                    except Exception as e:
                        st.error(f"创建异常: {str(e)}")
        else:
            st.info("👈 请在左侧提交日志进行分析")

# ==================== 页面4：自动化测试 ====================
elif page == "🧪 自动化测试":
    st.title("🧪 自动化测试")
    st.markdown("基于 Selenium 的自动化测试执行。选择场景并启动测试，查看实时结果和 AI 分析。")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("测试配置")
        # 默认使用后端内置的 Demo 测试站点（含登录/搜索/下单表单）
        _demo_url = f"{API_BASE}/demo"
        if "test_base_url" not in st.session_state:
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

        st.subheader("选择测试场景")
        scenarios = st.multiselect(
            "要执行的测试",
            options=["login", "search", "order"],
            default=["login", "search"],
            format_func=lambda x: {"login": "🔑 登录流程", "search": "🔍 搜索流程", "order": "🛒 下单流程"}[x],
        )

        auto_analyze = st.checkbox("失败时自动AI分析", value=True)
        sandbox = st.checkbox(
            "🧪 沙盒模式（无需Chrome，模拟执行）",
            value=False,
            help="开启后使用模拟的 Selenium 执行。关闭则驱动真实 Chrome 浏览器（默认关闭，使用真实浏览器）",
        )

        if st.button("▶ 开始测试", type="primary", disabled=not scenarios):
            mode_label = "沙盒模式" if sandbox else ("无头浏览器" if headless else "可见浏览器")
            with st.spinner(f"测试执行中...({mode_label})"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/v1/testing/run",
                        json={
                            "scenarios": scenarios,
                            "base_url": base_url,
                            "headless": headless,
                            "timeout_seconds": timeout,
                            "auto_analyze": auto_analyze,
                            "sandbox": sandbox,
                        },
                        timeout=600,
                    )
                    if resp.status_code == 200:
                        report = resp.json()
                        st.session_state.test_report = report
                        st.rerun()
                    else:
                        st.error(f"测试执行失败: {resp.text[:300]}")
                except requests.ConnectionError:
                    st.error(f"❌ 无法连接到后端 {API_BASE}")
                except Exception as e:
                    st.error(f"测试异常: {str(e)}")

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

            # 失败分析
            if report.get("failure_analysis"):
                st.subheader("🤖 AI 失败分析")
                for analysis in report["failure_analysis"]:
                    st.warning(analysis)
        else:
            st.info("👈 请在左侧配置并启动测试")

# ==================== 页面5：Agent 执行 ====================
elif page == "🤖 Agent 执行":
    st.title("🤖 Agent 执行")
    st.markdown("""
    **LangGraph Supervisor + ReAct 推理循环**

    输入任务描述，AI Agent 自动：
    1. 分析意图 → 2. 选择工具 → 3. 执行工具 → 4. 观察结果 → 5. 继续推理 → 6. 生成最终回答

    可用工具：`search_knowledge_base` | `parse_log_content` | `execute_test_scenario` | `create_jira_issue_tool` | `get_runtime_logs` | `get_system_status`
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("任务描述")

        # 快捷示例 — 一键填充常见任务
        st.caption("💡 点击示例快速填充：")
        ex_cols = st.columns(3)
        examples = {
            "刚才什么错误": "刚才后端运行出了什么错误？请读取运行日志并分析原因和解决方案。",
            "系统状态检查": "检查一下系统各组件状态是否正常，LLM、向量库、Chrome 是否可用。",
            "跑测试并分析": "执行登录和搜索场景的自动化测试，如果失败请分析原因并给出修复建议。",
        }
        for c, (label, text) in zip(ex_cols, examples.items()):
            if c.button(label, key=f"ex_{label}"):
                st.session_state.agent_task = text

        task = st.text_area(
            "用自然语言描述你想让 Agent 完成的任务",
            height=120,
            placeholder="例如：\n"
                       "• 刚才什么错误？读取运行日志分析一下\n"
                       "• 分析这个错误：TimeoutException: Page load timeout after 30s\n"
                       "• 执行登录测试，分析失败原因，确认是 bug 就创建 JIRA 缺陷单",
            key="agent_task",
        )

        max_iters = st.slider("ReAct 最大迭代次数", 1, 10, 5, key="agent_iters",
                              help="Agent 最多进行多少轮 思考→工具→观察 循环")

        task_type = st.selectbox(
            "任务类型（可选）",
            ["auto", "rag_query", "log_analysis", "test_execution", "jira_create"],
            format_func=lambda x: {
                "auto": "🤖 自动识别",
                "rag_query": "📚 知识库问答",
                "log_analysis": "📊 日志分析",
                "test_execution": "🧪 测试执行",
                "jira_create": "🎫 JIRA 创建"
            }[x],
            key="agent_task_type",
        )

        use_stream = st.checkbox("流式输出（SSE）", value=True, key="agent_stream",
                                 help="实时展示 Agent 的推理过程和工具调用")

        if st.button("🚀 执行 Agent 任务", type="primary", disabled=not task.strip(), key="agent_exec_btn"):
            # 清空上一次的结果
            st.session_state.agent_events = []
            st.session_state.pop("agent_result", None)

            with st.spinner("Agent 推理中..."):
                if use_stream:
                    # SSE 流式模式
                    stream_result = None
                    current_event_type = None
                    try:
                        resp = requests.post(
                            f"{API_BASE}/api/v1/agent/execute/stream",
                            json={"task": task, "max_iterations": max_iters, "task_type": task_type},
                            stream=True,
                            timeout=300,
                        )
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line:
                                current_event_type = None
                                continue
                            if line.startswith("event:"):
                                current_event_type = line[6:].strip()
                            elif line.startswith("data:"):
                                data_str = line[5:].strip()
                                if data_str:
                                    try:
                                        event_data = json.loads(data_str)
                                        st.session_state.agent_events.append({
                                            "event": current_event_type,
                                            "data": event_data,
                                        })
                                        if current_event_type == "final":
                                            stream_result = {
                                                "task_type": event_data.get("task_type", ""),
                                                "final_response": event_data.get("response", ""),
                                                "tool_calls_made": event_data.get("tool_calls", 0),
                                                "iterations": event_data.get("iterations", 0),
                                                "execution_time_ms": event_data.get("time_ms", 0),
                                            }
                                    except Exception as e:
                                        # SSE 事件解析失败时跳过该事件，不影响后续处理
                                        import logging as _logging
                                        _logging.getLogger("frontend").debug(f"SSE 事件解析跳过: {e}")
                    except Exception as e:
                        st.error(f"SSE 连接失败: {str(e)}")

                    if stream_result:
                        st.session_state.agent_result = stream_result
                        st.rerun()
                    elif st.session_state.agent_events:
                        # 流式有事件但无 final，仍展示已收集的推理过程
                        st.warning("流式未返回 final 事件，但已记录推理过程（见右侧）")
                    else:
                        st.info("流式未返回结果，切换到普通模式...")

                # 非流式模式（流式未启用或流式未成功时执行）
                if not use_stream or "agent_result" not in st.session_state:
                    try:
                        resp = requests.post(
                            f"{API_BASE}/api/v1/agent/execute",
                            json={"task": task, "max_iterations": max_iters, "task_type": task_type},
                            timeout=300,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.agent_result = data
                            st.rerun()
                        else:
                            st.error(f"执行失败 ({resp.status_code}): {resp.text[:300]}")
                    except requests.ConnectionError:
                        st.error(f"❌ 无法连接到后端 {API_BASE}")
                    except Exception as e:
                        st.error(f"执行异常: {str(e)}")

    with col2:
        st.subheader("执行结果")
        has_result = "agent_result" in st.session_state and st.session_state.agent_result
        has_events = bool(st.session_state.get("agent_events"))

        if has_result:
            result = st.session_state.agent_result

            # 统计卡片
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("任务类型", result.get("task_type", "N/A"))
            with m2:
                st.metric("工具调用", result.get("tool_calls_made", 0))
            with m3:
                st.metric("推理轮次", result.get("iterations", 0))
            with m4:
                st.metric("耗时", f"{result.get('execution_time_ms', 0):.0f}ms")

            # 错误信息
            if result.get("error"):
                st.error(f"执行错误: {result['error']}")

            # ---- 推理过程时间线（来自 SSE 事件） ----
            if has_events:
                with st.expander("🧠 推理过程时间线", expanded=True):
                    _render_agent_timeline(st.session_state.agent_events)

            # 最终回答
            st.markdown("### 💡 Agent 回答")
            final_response = result.get("final_response", "")
            if final_response:
                st.markdown(final_response)
            else:
                st.warning("Agent 未返回有效回答")

            # 任务信息
            with st.expander("📋 执行详情"):
                st.json({
                    "task_id": result.get("task_id", ""),
                    "task_type": result.get("task_type", ""),
                    "tool_calls_made": result.get("tool_calls_made", 0),
                    "iterations": result.get("iterations", 0),
                    "time_ms": result.get("execution_time_ms", 0),
                })
        elif has_events:
            # 仅有事件无最终结果（流式中断场景）
            st.info(f"已记录 {len(st.session_state.agent_events)} 个推理事件（未获得最终回答）")
            with st.expander("🧠 推理过程时间线", expanded=True):
                _render_agent_timeline(st.session_state.agent_events)
        else:
            st.info("👈 输入任务描述，点击执行按钮启动 Agent" + "\n\n" +
                    "**什么是 ReAct 循环？**\n\n" +
                    "Agent 不会一次性给出答案，而是：\n"
                    "1. 🧠 **Think** — 分析问题，决定要用什么工具\n"
                    "2. 🔧 **Act** — 调用工具（搜知识库/跑测试/读运行日志/查系统状态/创缺陷单）\n"
                    "3. 👁 **Observe** — 查看工具返回的结果\n"
                    "4. 🔄 重复以上步骤，直到获得足够信息给出最终答案")


# ==================== 辅助函数：渲染 Agent 推理时间线 ====================

def _render_agent_timeline(events: list[dict]):
    """将 SSE 事件列表渲染为可视化的推理时间线

    事件类型：
    - start       : 任务开始
    - node_start  : 进入图节点（supervisor / rag_node / execute_tools ...）
    - tool_call   : Agent 调用工具
    - node_end    : 节点执行完毕
    - final       : 最终回答
    - error       : 错误
    - done        : 流结束
    """
    if not events:
        st.caption("（无推理事件）")
        return

    node_icons = {
        "supervisor": "🎯",
        "rag_node": "📚",
        "analysis_node": "📊",
        "test_node": "🧪",
        "jira_node": "🎫",
        "execute_tools": "🔧",
        "format_output": "📝",
    }

    step_idx = 0
    for ev in events:
        etype = ev.get("event", "")
        data = ev.get("data", {}) or {}

        if etype == "start":
            st.markdown(f"**▶ 任务开始** — `{data.get('task', '')[:80]}`")
            st.caption(f"task_id: {data.get('task_id', '')} | 最大迭代: {data.get('max_iterations', '')}")

        elif etype == "node_start":
            step_idx += 1
            node = data.get("node", "")
            icon = node_icons.get(node, "⚙️")
            it = data.get("iteration", 0)
            label = {
                "supervisor": "意图分类",
                "rag_node": "RAG 问答推理",
                "analysis_node": "日志分析推理",
                "test_node": "测试执行推理",
                "jira_node": "JIRA 创建推理",
                "execute_tools": "执行工具",
                "format_output": "格式化输出",
            }.get(node, node)
            st.markdown(f"**{icon} 步骤 {step_idx}：{label}**" + (f" · 第 {it} 轮" if it else ""))

        elif etype == "tool_call":
            tool = data.get("tool", "unknown")
            args = data.get("args", {})
            args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else "(无参数)"
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ 🔧 调用工具 **`{tool}`**({args_str})")

        elif etype == "node_end":
            nxt = data.get("next_action", "")
            if nxt:
                st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;→ 下一步: `{nxt}`")

        elif etype == "final":
            st.markdown(
                f"**✅ 完成** — 工具调用 {data.get('tool_calls', 0)} 次 | "
                f"迭代 {data.get('iterations', 0)} 轮 | "
                f"耗时 {data.get('time_ms', 0):.0f}ms"
            )

        elif etype == "error":
            st.error(f"❌ 错误: {data.get('message', '')}")

        elif etype == "done":
            st.caption("— 流结束 —")
