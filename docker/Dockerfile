# ==================== 多阶段构建 Dockerfile ====================
# Stage 1: 依赖安装
FROM python:3.12-slim AS builder

WORKDIR /app

# 安装系统依赖（Chrome 和 Chromium 用于 Selenium）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 安装 Chromium（Selenium headless 模式所需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# ==================== Stage 2: 运行镜像 ====================
FROM python:3.12-slim

WORKDIR /app

# 安装 Chromium 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# 设置 Chromium 环境变量（供 Selenium 使用）
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# 复制项目代码
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY scripts/ ./scripts/

# 创建非 root 用户运行
RUN useradd -m -s /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
