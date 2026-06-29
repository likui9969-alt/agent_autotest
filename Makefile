.PHONY: install test lint clean run docker-up docker-build help

help: ## 显示帮助信息
	@printf "\033[36m%-20s\033[0m %s\n" "目标" "说明"
	@printf "\033[36m%-20s\033[0m %s\n" "------" "----"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖（虚拟环境中）
	pip install -r requirements.txt

run: ## 启动开发服务器（热重载）
	uvicorn backend.main:app --reload --port 8001

run-prod: ## 启动生产服务器
	uvicorn backend.main:app --host 0.0.0.0 --port 8000

test: ## 运行全部测试（排除 Selenium）
	python -m pytest tests/ -v --tb=short --ignore=tests/test_selenium -p no:cacheprovider

test-all: ## 运行全部测试（含 Selenium）
	python -m pytest tests/ -v --tb=short -p no:cacheprovider

test-rag: ## 运行 RAG 相关测试
	python -m pytest tests/test_rag/ -v --tb=short

test-agent: ## 运行 Agent 相关测试
	python -m pytest tests/test_agent/ -v --tb=short

test-coverage: ## 运行测试并生成覆盖率报告
	python -m pytest tests/ -v --tb=short --cov=backend --cov-report=term --cov-report=html \
		--ignore=tests/test_selenium -p no:cacheprovider

lint: ## 代码风格检查
	ruff check backend/ tests/ || true

format: ## 自动格式化代码
	ruff check --fix backend/ tests/ || true

typecheck: ## 类型检查（mypy）
	python -m mypy backend/ --ignore-missing-imports || true

clean: ## 清理缓存和临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/ 2>/dev/null || true

docker-build: ## 构建 Docker 镜像
	docker compose -f docker/docker-compose.yml build

docker-up: ## 启动 Docker 容器（后台）
	docker compose -f docker/docker-compose.yml up -d

docker-down: ## 停止 Docker 容器
	docker compose -f docker/docker-compose.yml down

docker-logs: ## 查看容器日志
	docker compose -f docker/docker-compose.yml logs -f

seed: ## 向知识库填充示例数据
	python scripts/seed_knowledge_base.py

precommit: lint test ## 提交前检查（lint + test）
