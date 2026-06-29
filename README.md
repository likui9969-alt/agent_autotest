# AI 研发效能智能体

> 基于 RAG + LangGraph Multi-Agent 的自动化测试与故障分析系统

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)](https://langchain-ai.github.io/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)

## 简介

面向测试工程师的智能助手，能自动分析测试日志、检索历史故障案例、执行自动化测试并生成缺陷报告。

核心能力：
- 管理测试知识库（日志、缺陷单、技术文档），支持上传、检索、增量索引
- 基于 RAG 的智能问答，显示引用来源和相关度分数
- LangGraph 多 Agent 协作：日志分析 → 循环检测 → 测试生成 → JIRA 创建，全链路自动执行
- LLM 多 Provider 支持：DashScope / OpenAI / Ollama，可配置回退链
- Selenium 自动化测试 + AI 失败分析 + 测试用例自动生成

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit 前端（7 页面）                  │
│  知识库 │ 智能问答 │ 日志分析 │ 自动化测试 │ 测试用例 │ Agent │ 对话 │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/REST
┌───────────────────────────▼─────────────────────────────────┐
│                        FastAPI 后端                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         LangGraph Supervisor Agent                   │    │
│  │  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐  │    │
│  │  │ 日志分析   │ │ 循环检测  │ │ 测试生成  │ │ JIRA │  │    │
│  │  │ Agent     │ │          │ │ Agent    │ │ Agent│  │    │
│  │  └─────┬─────┘ └──────────┘ └─────┬────┘ └──┬───┘  │    │
│  │        │   Memory (上下文+短期记忆)  │         │       │    │
│  └────────┼──────────────────────────┼─────────┼───────┘    │
│           │                          │         │            │
│  ┌────────▼────┐  ┌─────────────────▼┐  ┌────▼─────┐      │
│  │   RAG 管线   │  │  Selenium 引擎    │  │ JIRA API │      │
│  │ Chroma 向量库│  │  场景: 登录/搜索/  │  │          │      │
│  └─────────────┘  │  下单/自定义       │  └──────────┘      │
│                   └──────────────────┘                     │
│                                                             │
│  LLM: DashScope / OpenAI / Ollama （可配置回退）             │
└─────────────────────────────────────────────────────────────┘
```

### 关键模块

| 模块 | 说明 |
|------|------|
| `backend/agent/` | LangGraph Supervisor 图调度，含 Memory、LoopDetector、TaskRegistry |
| `backend/llm/providers/` | LLM 工厂模式，统一接口对接 DashScope / OpenAI / Ollama |
| `backend/rag/` | 文档加载 → 切片 → 嵌入 → Chroma 存储 → 检索 |
| `backend/selenium_driver/` | Selenium WebDriver 封装，支持多场景复用 |
| `backend/db/` | SQLite 持久化（报告 + 测试用例） |
| `frontend/pages/` | Streamlit 7 页面，每个页面独立模块 |

---

## 快速开始

### 前置条件

- Python 3.12+
- Chrome（Selenium 测试用）
- LLM API Key（DashScope / OpenAI 任选，或本地 Ollama）

### 安装

```bash
git clone https://github.com/likui9969-alt/agent_autotest.git
cd agent_one_test

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key 和模型配置（见 `.env.example` 中各字段说明）。

### 初始化知识库

```bash
python scripts/seed_knowledge_base.py
```

### 启动

```bash
# 后端（端口 8000）
uvicorn backend.main:app --reload --port 8000

# 前端（新终端，端口 8501）
streamlit run frontend/app.py
```

API 文档：http://localhost:8000/docs

---

## Docker 部署

```bash
docker-compose -f docker/docker-compose.yml up -d
docker-compose -f docker/docker-compose.yml logs -f
docker-compose -f docker/docker-compose.yml down
```

---

## 项目结构

```
agentone_test/
├── backend/
│   ├── main.py                 # FastAPI 入口
│   ├── config/                 # 配置管理（Pydantic Settings）
│   ├── models/                 # Pydantic 数据模型
│   ├── llm/
│   │   ├── client.py           # LLM 客户端（重试 + 熔断）
│   │   └── providers/          # 多 Provider 工厂
│   │       ├── dashscope.py
│   │       ├── openai.py
│   │       ├── openai_compatible.py
│   │       ├── ollama.py
│   │       └── factory.py
│   ├── agent/
│   │   ├── graph.py            # LangGraph Supervisor 图
│   │   ├── log_analyzer.py     # 日志分析 Agent
│   │   ├── loop_detector.py    # 循环检测
│   │   ├── test_generator.py   # 测试用例生成
│   │   ├── jira_creator.py     # JIRA Agent
│   │   ├── memory/             # Agent 记忆模块
│   │   ├── task_registry.py    # 任务注册表
│   │   └── state.py            # Agent 共享状态
│   ├── rag/                    # RAG 管线
│   ├── db/                     # SQLite 持久化
│   ├── api/routes/             # API 路由
│   │   ├── knowledge.py        # 知识库管理
│   │   ├── rag.py              # 智能问答
│   │   ├── analysis.py         # 日志分析
│   │   ├── testing.py          # 自动化测试
│   │   ├── test_cases.py       # 测试用例管理
│   │   ├── agents.py           # Agent 执行
│   │   └── jira.py             # JIRA 集成
│   └── selenium_driver/        # Selenium 引擎
│       └── scenarios/          # 登录 / 搜索 / 下单 / 自定义
├── frontend/
│   ├── app.py                  # Streamlit 入口
│   ├── pages/                  # 7 个页面模块
│   └── utils/                  # API 客户端 / 状态管理 / 样式
├── data/                       # 运行时数据（已 gitignore）
├── scripts/                    # 辅助脚本
├── docker/                     # Dockerfile + Compose
├── tests/                      # 测试
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/v1/knowledge/stats` | 知识库统计 |
| POST | `/api/v1/knowledge/upload` | 上传文档并索引 |
| POST | `/api/v1/knowledge/rebuild` | 重建向量库 |
| POST | `/api/v1/knowledge/incremental` | 增量索引目录 |
| POST | `/api/v1/rag/query` | RAG 智能问答 |
| POST | `/api/v1/analysis/log` | 日志分析 |
| POST | `/api/v1/testing/run` | 自动化测试执行 |
| POST | `/api/v1/test-cases/generate` | AI 生成测试用例 |
| GET | `/api/v1/test-cases/` | 测试用例列表 |
| GET | `/api/v1/test-cases/{id}` | 测试用例详情 |
| DELETE | `/api/v1/test-cases/{id}` | 删除测试用例 |
| GET | `/api/v1/test-cases/export/csv` | 导出 CSV |
| POST | `/api/v1/agent/run` | LangGraph Agent 执行 |
| POST | `/api/v1/jira/create` | 创建 JIRA 缺陷单 |

---

## 简历描述

> **AI 研发效能智能体系统** | 独立开发 | 2025.06
>
> - 设计 LangGraph Multi-Agent 架构，Supervisor 调度日志分析、循环检测、测试生成、JIRA 创建四个 Agent 协作完成端到端故障处理
> - 实现 LLM 多 Provider 工厂（DashScope / OpenAI / Ollama），支持指数退避重试和熔断保护，可配置回退链保证服务可用性
> - 基于 RAG（LangChain + Chroma）构建测试知识库，支持文档上传、向量化检索、增量索引，用于历史故障相似案例匹配
> - 使用 FastAPI + Streamlit 前后端分离架构，7 个独立页面覆盖知识库管理、智能问答、日志分析、自动化测试、测试用例管理、Agent 执行、对话
> - Selenium WebDriver 封装多场景自动化测试，AI 失败分析；SQLite 持久化报告和测试用例；Docker 容器化部署

---

## 技术栈

- **后端**: Python 3.12 / FastAPI / Pydantic
- **前端**: Streamlit
- **Agent**: LangGraph / LangChain
- **LLM**: DashScope (DeepSeek-V3) / OpenAI / Ollama
- **RAG**: LangChain + Chroma
- **自动化**: Selenium WebDriver
- **部署**: Docker / Docker Compose

---

## 提交前注意

`.gitignore` 已排除以下敏感/运行时文件：

```
.env                # 含 API Key，不可提交
chromedriver.exe    # 浏览器驱动二进制
.mcp.json           # 本地 MCP 配置
.claude/            # Claude 本地设置
data/logs/*         # 运行时日志
data/chroma/*       # Chroma 向量数据库
data/screenshots/   # 测试截图
data/reports.db     # 运行时数据库
data/backups/       # Chroma 备份
```

新增敏感文件时记得补充 `.gitignore`。

## License

MIT
