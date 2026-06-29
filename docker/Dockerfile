# ==================== 多阶段构建 Dockerfile ====================
# Stage 1: 依赖安装（编译重型依赖，减少最终镜像层）
FROM python:3.12-slim AS builder

WORKDIR /app

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖到独立前缀
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==================== Stage 2: 运行镜像 ====================
FROM python:3.12-slim

WORKDIR /app

# 安装 Chromium 运行时依赖 + curl（健康检查）
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置 Chromium 环境变量（供 Selenium 使用）
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV PYTHONUNBUFFERED=1
ENV LOG_FORMAT=json

# 从 builder 复制已安装的 Python 包
COPY --from=builder /install /usr/local

# 复制项目代码
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# 创建非 root 用户运行
RUN useradd -m -s /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查 — 验证后端核心接口
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
