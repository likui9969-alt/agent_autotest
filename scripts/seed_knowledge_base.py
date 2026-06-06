"""
知识库初始化脚本
批量导入示例文档，快速搭建可演示的知识库

用法：python scripts/seed_knowledge_base.py
"""
import sys
import os
from pathlib import Path

# 设置控制台输出编码为 UTF-8（解决 Windows GBK 编码问题）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import get_settings
from backend.rag.pipeline import RAGPipeline


# 示例故障案例数据
SAMPLE_CASES = [
    {
        "filename": "case_001_login_timeout.txt",
        "content": """# 故障案例 001：登录页面加载超时

## 问题描述
用户在点击登录按钮后，页面加载超过 30 秒无响应，最终抛出 TimeoutException。

## 异常信息
TimeoutException: Page load timeout after 30000ms
  File "test_login.py", line 45, in test_login
    driver.get("https://example.com/login")

## 根本原因
后端认证服务在高峰期响应缓慢，数据库连接池耗尽导致请求排队。

## 解决方案
1. 增加数据库连接池大小（从 20 增加到 50）
2. 为登录接口添加缓存层
3. 调整 Selenium 超时时间为 60 秒
4. 在 Nginx 层面添加请求限流

## 影响范围
所有需要登录的功能模块

## 严重等级
高""",
    },
    {
        "filename": "case_002_element_not_found.txt",
        "content": """# 故障案例 002：搜索结果元素定位失败

## 问题描述
自动化测试在执行搜索操作后，无法找到搜索结果列表元素。

## 异常信息
NoSuchElementException: Unable to locate element: {"method":"css selector","selector":".search-results"}
  File "test_search.py", line 78, in verify_search_results
    results = driver.find_element(By.CSS_SELECTOR, ".search-results")

## 根本原因
前端近期上线了新版本，搜索结果区域的 CSS 类名从 .search-results 变更为 .results-container。

## 解决方案
1. 更新 Selenium 定位器为新的 CSS 选择器
2. 建立前端变更通知机制，UI 变更时同步更新测试脚本
3. 使用更鲁棒的定位策略（data-testid 属性替代 CSS 类名）

## 影响范围
所有搜索相关测试用例

## 严重等级
中""",
    },
    {
        "filename": "case_003_connection_refused.txt",
        "content": """# 故障案例 003：服务连接被拒绝

## 问题描述
测试环境自动化测试批量运行时，出现 ConnectionError。

## 异常信息
ConnectionError: HTTPConnectionPool(host='test-server', port=8080): Max retries exceeded
  File "base_test.py", line 23, in setup
    response = requests.get(f"{BASE_URL}/health")

## 根本原因
Docker 容器中的测试服务未正确启动，健康检查端口 8080 在容器启动 30 秒后才可用。

## 解决方案
1. 在 docker-compose.yml 中添加 healthcheck 配置
2. 测试脚本增加重试逻辑（最多 3 次，间隔 5 秒）
3. 添加服务启动等待脚本

## 影响范围
所有依赖该服务的测试用例

## 严重等级
高""",
    },
    {
        "filename": "case_004_assertion_error.txt",
        "content": """# 故障案例 004：断言失败 — 订单金额校验不通过

## 问题描述
下单流程测试中，订单确认页面的总金额与预期值不匹配。

## 异常信息
AssertionError: Expected order total 299.00 but got 329.00
  File "test_order.py", line 156, in verify_order_total
    assert actual_total == expected_total

## 根本原因
系统新增了配送费计算逻辑（满 299 免运费），测试用例中的预期值未更新。

## 解决方案
1. 更新测试用例的预期金额计算逻辑
2. 将金额相关断言改为范围校验（允许 ±5 元误差）
3. 测试数据与业务规则解耦，从配置中读取阈值

## 影响范围
下单流程测试用例

## 严重等级
中""",
    },
    {
        "filename": "case_005_sql_error.txt",
        "content": """# 故障案例 005：数据库写入异常

## 问题描述
高并发测试时出现 SQL 写入异常。

## 异常信息
OperationalError: (1040, 'Too many connections')
  File "db_utils.py", line 89, in execute_query
    cursor.execute(sql, params)

## 根本原因
1. 数据库最大连接数设置为 100，并发测试时超过限制
2. 部分代码未正确关闭数据库连接，造成连接泄漏

## 解决方案
1. 将 max_connections 从 100 增加到 500
2. 使用连接池（SQLAlchemy pool_size=20, max_overflow=40）
3. 在 finally 块中确保连接关闭
4. 添加连接超时自动回收机制

## 影响范围
所有涉及数据库操作的测试用例

## 严重等级
高""",
    },
]


def main():
    """初始化知识库"""
    print("=" * 60)
    print("  知识库初始化脚本")
    print("=" * 60)

    settings = get_settings()
    docs_dir = Path(settings.get_upload_dir())
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成示例文档
    print(f"\n📝 生成示例文档到 {docs_dir} ...")
    created = 0
    for case in SAMPLE_CASES:
        filepath = docs_dir / case["filename"]
        if not filepath.exists():
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(case["content"])
            print(f"   ✓ 创建: {case['filename']}")
            created += 1
        else:
            print(f"   ⊘ 已存在: {case['filename']}")

    print(f"\n   共创建 {created} 个示例文档")

    # 2. 索引文档
    print("\n🔨 开始索引文档到 Chroma 向量库 ...")
    pipeline = RAGPipeline()
    chunk_count = pipeline.ingest_directory(str(docs_dir))

    print(f"\n✅ 知识库初始化完成！")
    print(f"   文档数: {len(SAMPLE_CASES)}")
    print(f"   向量块数: {chunk_count}")
    print(f"   Chroma 目录: {settings.get_chroma_dir()}")

    # 3. 显示统计
    stats = pipeline.stats()
    print(f"\n📊 知识库统计:")
    print(f"   集合名称: {stats['collection_name']}")
    print(f"   总块数: {stats['total_chunks']}")


if __name__ == "__main__":
    main()
