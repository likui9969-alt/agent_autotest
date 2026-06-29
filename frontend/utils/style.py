"""全局 CSS 样式。"""
from __future__ import annotations

GLOBAL_CSS = """
<style>
/* 统一按钮内边距与最小高度，避免文字换行后过于拥挤 */
.stButton > button {
    min-height: 2.5rem;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    font-weight: 500;
}

/* 模板按钮组：增加按钮之间的水平/垂直间距 */
.tpl-btn {
    margin-bottom: 0.4rem;
}

/* 步骤编辑器表头 */
.step-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: #888;
    margin-bottom: -0.5rem;
    padding-top: 0.5rem;
}

/* 步骤行卡片：增加上下间距和视觉分隔 */
.step-row {
    padding: 0.6rem 0.4rem;
    margin: 0.4rem 0;
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 0.5rem;
    background-color: rgba(128, 128, 128, 0.05);
}

/* 示例按钮组增加间距 */
.example-btn {
    margin-bottom: 0.4rem;
}

/* 左侧配置区各个小节增加间距 */
.config-section {
    margin-bottom: 1.2rem;
}

/* 模板按钮增加垂直间距 */
.tpl-btn button {
    margin: 0.3rem 0;
}

/* 步骤编辑器列之间增加水平间距 */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    padding-left: 0.25rem;
    padding-right: 0.25rem;
}

/* 移动端：步骤编辑器列太窄时让输入框自动占满 */
@media (max-width: 768px) {
    .step-row [data-testid="stVerticalBlock"] > div {
        width: 100% !important;
    }
}
</style>
"""
