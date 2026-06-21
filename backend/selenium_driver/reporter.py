"""
测试报告生成器模块
将测试结果导出为 HTML 或 JSON 格式
"""
import json
import logging
from pathlib import Path
from datetime import datetime

from backend.models.testing import TestReport, TestCaseResult, TestStatus
from backend.config.settings import PROJECT_ROOT

logger = logging.getLogger("ai_rd_agent")


class TestReporter:
    """测试报告生成器

    支持格式：
    - JSON：机器可读，适合后续处理或 API 返回
    - HTML：人类可读，适合邮件发送或浏览器查看

    使用示例：
        reporter = TestReporter()
        report_path = reporter.generate_html(report)
        report_json = reporter.generate_json(report)
    """

    def __init__(self, output_dir: str | None = None):
        """
        Args:
            output_dir: 报告输出目录（不传则使用 data/test_reports/）
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = PROJECT_ROOT / "data" / "test_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json(self, report: TestReport) -> str:
        """生成 JSON 格式测试报告

        Args:
            report: 测试报告对象

        Returns:
            JSON 文件的保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{report.report_id}_{timestamp}.json"
        filepath = self.output_dir / filename

        # 将 Pydantic 模型转为字典
        report_dict = report.model_dump(mode="json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"JSON 测试报告已保存: {filepath}")
        return str(filepath)

    def generate_html(self, report: TestReport) -> str:
        """生成 HTML 格式测试报告

        Args:
            report: 测试报告对象

        Returns:
            HTML 文件的保存路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{report.report_id}_{timestamp}.html"
        filepath = self.output_dir / filename

        html_content = self._build_html(report)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML 测试报告已保存: {filepath}")
        return str(filepath)

    def _build_html(self, report: TestReport) -> str:
        """构建 HTML 报告内容"""
        # 状态对应的 CSS 样式
        status_colors = {
            TestStatus.PASSED: "#28a745",
            TestStatus.FAILED: "#dc3545",
            TestStatus.SKIPPED: "#ffc107",
            TestStatus.ERROR: "#fd7e14",
        }

        # 构建每个用例的 HTML 行
        rows_html = ""
        for result in report.results:
            color = status_colors.get(result.status, "#6c757d")
            steps_html = ""
            for step in result.steps:
                step_color = status_colors.get(step.status, "#6c757d")
                steps_html += f"""
                <tr>
                    <td style="padding:4px 12px;">{step.step_name}</td>
                    <td style="padding:4px 12px;color:{step_color};">{step.status.value}</td>
                    <td style="padding:4px 12px;">{step.duration_ms:.0f}ms</td>
                    <td style="padding:4px 12px;color:#dc3545;font-size:0.9em;">{step.error_message}</td>
                </tr>"""

            rows_html += f"""
            <tr style="border-bottom:1px solid #ddd;">
                <td style="padding:8px 12px;font-weight:bold;">{result.scenario}</td>
                <td style="padding:8px 12px;color:{color};font-weight:bold;">{result.status.value}</td>
                <td style="padding:8px 12px;">{result.duration_ms:.0f}ms</td>
                <td style="padding:8px 12px;color:#dc3545;">{result.error_message[:200]}</td>
            </tr>
            <tr>
                <td colspan="4" style="padding:0;">
                    <table style="width:100%;background:#fafafa;">
                        {steps_html}
                    </table>
                </td>
            </tr>"""

        # 失败分析 HTML
        analysis_html = ""
        if report.failure_analysis:
            items = "".join(f"<li>{item}</li>" for item in report.failure_analysis)
            analysis_html = f"""
            <div style="margin-top:20px;padding:15px;background:#fff3cd;border-radius:5px;">
                <h3 style="color:#856404;">⚠ 失败分析</h3>
                <ul>{items}</ul>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>自动化测试报告 - {report.report_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f0f2f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ padding: 15px 25px; border-radius: 8px; color: white; font-size: 1.1em; }}
        .stat.total {{ background: #4a90d9; }}
        .stat.passed {{ background: #28a745; }}
        .stat.failed {{ background: #dc3545; }}
        .stat.rate {{ background: #6f42c1; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #4a90d9; color: white; padding: 10px 12px; text-align: left; }}
        td {{ padding: 8px 12px; }}
        .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #eee; color: #999; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 自动化测试报告</h1>

        <div class="summary">
            <div class="stat total">总计: {report.total_scenarios}</div>
            <div class="stat passed">通过: {report.passed_count}</div>
            <div class="stat failed">失败: {report.failed_count}</div>
            <div class="stat rate">通过率: {report.pass_rate:.0%}</div>
        </div>

        <p>📋 报告ID: {report.report_id} | 🔗 目标URL: {report.base_url} | ⏱ 执行时间: {report.executed_at.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <table>
            <tr>
                <th>测试场景</th>
                <th>状态</th>
                <th>耗时</th>
                <th>错误信息</th>
            </tr>
            {rows_html}
        </table>

        {analysis_html}

        <div class="footer">
            AI-Driven 研发效能智能体 — 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""
