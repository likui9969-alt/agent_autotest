# 🤖 AI-Driven 研发效能智能体

> 基于 RAG 的自动化测试与故障分析系统

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io)
[![Chroma](https://img.shields.io/badge/Chroma-0.5-orange.svg)](https://www.trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)

## 📖 项目简介

AI-Driven 研发效能智能体是一个面向测试工程师与研发团队的智能助手系统。系统基于 **RAG（检索增强生成）** 技术，结合 **LangChain** 和 **阿里云百炼大模型**，能够：

- 📚 管理历史测试知识库（测试日志、缺陷单、技术文档）
- 🔍 智能检索历史相似故障案例
- 📊 自动分析测试日志，识别异常根因
- 🧪 执行 Selenium 自动化测试
- 🎫 自动创建 JIRA 缺陷单
- 💬 提供基于知识库的智能问答

---

## 🏗 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit 前端                        │
│   📚知识库管理 │ 💬智能问答 │ 📊日志分析 │ 🧪自动化测试    │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI 后端                           │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │ 知识库API │ RAG API  │ 分析API  │ 测试API │ JIRA API│  │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬───┘  │
│       │          │          │          │          │      │
│  ┌────▼──┐  ┌───▼───┐  ┌───▼───┐  ┌───▼───┐  ┌──▼───┐  │
│  │  RAG  │  │Agent  │  │ 日志   │  │Selenium│  │JIRA  │  │
│  │ 管线  │  │调度器 │  │分析Agent│ │ 引擎  │  │集成   │  │
│  └───┬───┘  └───────┘  └───┬───┘  └───────┘  └──────┘  │
│      │                     │                             │
└──────┼─────────────────────┼─────────────────────────────┘
       │                     │
┌──────▼──────┐     ┌────────▼────────┐
│   Chroma    │     │  阿里云百炼 LLM  │
│  向量数据库 │     │  (DashScope API) │
└─────────────┘     └─────────────────┘
```

### 核心组件

| 组件 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI | 高性能异步 Web 框架，自动生成 Swagger 文档 |
| 前端 | Streamlit | 快速构建数据应用的 Python 框架 |
| LLM | 阿里云百炼 (DashScope) | DeepSeek-V3 对话模型 + text-embedding-v3 嵌入模型 |
| RAG 框架 | LangChain | 文档加载、切割、检索管线 |
| 向量数据库 | Chroma | 轻量级嵌入式向量存储 |
| 自动化测试 | Selenium | Chrome headless WebDriver |
| 缺陷管理 | JIRA REST API | 自动创建和更新缺陷单 |

---

## 🚀 快速开始

### 前置条件

- Python 3.12+
- Chrome 浏览器（用于 Selenium 测试）
- 阿里云百炼 API Key（[免费申请](https://bailian.console.aliyun.com/)）

### 1. 克隆项目

```bash
git clone <repo-url>
cd agentone_test
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

编辑 `.env` 文件，填入你的 API Key：

```env
DASHSCOPE_API_KEY=sk-your-api-key-here
DASHSCOPE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=deepseek-v3
EMBEDDING_MODEL=text-embedding-v3
```

### 5. 初始化知识库

```bash
python scripts/seed_knowledge_base.py
```

这将自动生成 5 个示例故障案例文档并索引到 Chroma 向量库。

### 6. 启动后端服务

```bash
uvicorn backend.main:app --reload --port 8000
```

访问 http://localhost:8000/docs 查看 Swagger API 文档。

### 7. 启动前端（新终端）

```bash
streamlit run frontend/app.py
```

访问 http://localhost:8501 进入前端界面。

---

## 🐳 Docker 部署

```bash
# 构建并启动所有服务
docker-compose -f docker/docker-compose.yml up -d

# 查看日志
docker-compose -f docker/docker-compose.yml logs -f

# 停止服务
docker-compose -f docker/docker-compose.yml down
```

服务端口：
- 后端 API: http://localhost:8000
- API 文档: http://localhost:8000/docs
- 前端: http://localhost:8501

---

## 📂 项目结构

```
agentone_test/
├── backend/                        # 后端源码
│   ├── main.py                     # FastAPI 入口
│   ├── config/                     # 配置管理（Pydantic Settings）
│   ├── models/                     # 数据模型（Pydantic）
│   ├── llm/                        # LLM 客户端（百炼 API）
│   ├── rag/                        # RAG 管线（加载/切割/嵌入/存储/检索）
│   ├── api/                        # API 路由（7组）
│   │   └── routes/                 # health/knowledge/rag/analysis/testing/jira
│   ├── agent/                      # Agent 模块（日志分析/测试执行/JIRA）
│   ├── selenium_driver/            # Selenium 自动化测试引擎
│   │   └── scenarios/              # 测试场景（登录/搜索/下单）
│   └── utils/                      # 通用工具
├── frontend/                       # Streamlit 前端
│   └── app.py                      # 4 页面应用
├── data/                           # 持久化数据
│   ├── docs/                       # 上传文档
│   ├── logs/                       # 日志文件
│   └── chroma/                     # Chroma 向量库
├── scripts/                        # 辅助脚本
│   └── seed_knowledge_base.py      # 知识库初始化
├── docker/                         # Docker 部署
│   ├── Dockerfile
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── tests/                          # 单元测试
├── requirements.txt
├── .env                            # 环境变量
└── README.md
```

---

## 🔌 API 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/v1/knowledge/stats` | 知识库统计 |
| POST | `/api/v1/knowledge/upload` | 上传文档 |
| POST | `/api/v1/knowledge/rebuild` | 重建向量库 |
| POST | `/api/v1/rag/query` | RAG 智能问答 |
| POST | `/api/v1/analysis/log` | 日志分析 |
| POST | `/api/v1/testing/run` | 自动化测试执行 |
| POST | `/api/v1/jira/create` | JIRA 缺陷创建 |

---

## 🖥 功能介绍

### 📚 知识库管理
- 支持上传 txt / pdf / docx 格式文档
- 自动切割（chunk_size=1000, overlap=200）并向量化
- 一键重建向量库

### 💬 智能问答
- 基于 RAG 的知识库问答
- 支持相似度检索和 MMR 检索
- 显示引用来源和相关度分数

### 📊 日志分析
- 上传或粘贴测试日志
- 自动识别 12 种常见异常类型
- 检索历史相似案例
- 生成结构化故障分析报告（摘要/原因/案例/建议）

### 🧪 自动化测试
- Selenium Chrome 自动化执行
- 支持登录/搜索/下单 3 种场景
- 失败自动调用 AI 分析
- 生成 HTML 测试报告

---

## 📝 简历描述示例

> **AI-Driven 研发效能智能体系统** | 独立开发 | 2025.06
>
> - 设计并实现了一套基于 RAG 技术的自动化测试与故障分析系统，集成 LangChain + Chroma 向量数据库
> - 使用阿里云百炼 DeepSeek-V3 大模型，实现了智能日志分析 Agent，可自动识别 12 种常见异常类型并给出根因分析和修复建议
> - 基于 FastAPI 构建 RESTful 后端，Streamlit 构建交互式前端，支持知识库管理、智能问答、日志分析、自动化测试四大功能模块
> - 集成 Selenium 实现 Web 自动化测试，失败时自动触发 AI 分析；支持 JIRA 缺陷单自动创建
> - 项目采用 Docker 容器化部署，前后端分离架构，可通过 docker-compose 一键启动

---

## 🛠 技术栈

- **后端**: Python 3.12 / FastAPI / Pydantic
- **前端**: Streamlit
- **LLM**: 阿里云百炼 API (DeepSeek-V3 / text-embedding-v3)
- **RAG**: LangChain / Chroma
- **自动化**: Selenium WebDriver
- **部署**: Docker / Docker Compose

---

## 📄 License

MIT License
