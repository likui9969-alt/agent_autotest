"""
数据库持久化模块（SQLite 简单实现）

用于持久化测试报告、Agent 执行记录等。
"""
from backend.db.reports import TestReportStore

__all__ = ["TestReportStore"]
