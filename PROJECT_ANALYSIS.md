# Open Deep Research — 项目完整分析报告

> 分析日期：2026-06-12 | 当前分支：`feature/bendicaozuo` | 版本：0.0.16

---

## 一、项目概览

**Open Deep Research** 是一个深度研究 Agent 操作系统（Agent OS），基于 GitHub 开源项目 `open_deep_research` 二次开发。项目定位为 **信息差 Agent OS 闭环平台**，实现从信息发现（Feed）→ 研究分析（Agent Runtime / Deep Research）→ 产物生成（Artifact）→ 记忆与技能沉淀（Memory / Skill）的完整闭环。

核心能力：
- **多源信息 Feed**：GitHub、ArXiv、Tavily、SerpAPI、DuckDuckGo、RSS → 智能卡片推荐
- **多智能体协作**：Planner → Router → 各领域 Agent（Research/RAG/Memory/Tool/Skill/Artifact）→ Final Response
- **RAG 知识库**：文档上传 → 结构化分块 → 向量+BM25 混合检索 → 问答
- **分级记忆系统**：Working → Episodic → Semantic 三级记忆 + Qdrant 向量搜索 + Neo4j 知识图谱
- **MCP 工具生态**：内置 email.send、local_file.read/write 等工具 + 审批流
- **Skill 复用机制**：从 Agent 工作流中提取可复用 Skill 草稿

---

## 二、技术栈

| 层级 | 技术选型 |
|------|---------|
| **Web 框架** | FastAPI + Uvicorn |
| **Agent 编排** | 自研 Runtime（非标准 LangGraph，自定义节点图） |
| **LLM** | 阿里云 DashScope（Qwen 全系：qwen3.6-max-preview / qwen3.6-plus / qwen3.7-plus） |
| **向量数据库** | Qdrant（文档 + 记忆向量） |
| **图数据库** | Neo4j（记忆图谱 + 项目知识图谱） |
| **关系数据库** | PostgreSQL（SQLAlchemy ORM + Alembic） |
| **缓存** | Redis |
| **前端** | React + Vite + TypeScript（未在此仓库） |
| **依赖管理** | UV（uv.lock） |
| **LLM SDK** | LangChain（ChatOpenAI 兼容模式调用阿里云） |

---

## 三、架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (React/Vite)                     │
└──────────────────┬──────────────────────────────────────┘
                   │ REST API (SSE streaming)
┌──────────────────▼──────────────────────────────────────┐
│                FastAPI (src/web_app/main.py)              │
│  ├─ /api/v1/auth       ├─ /api/v1/feed                  │
│  ├─ /api/v1/agent      ├─ /api/v1/research              │
│  ├─ /api/v1/memory     ├─ /api/v1/skills                │
│  ├─ /api/v1/documents  ├─ /api/v1/artifacts             │
│  ├─ /api/v1/approvals  ├─ /api/v1/mcp                   │
│  └─ /api/v1/profile                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Services Layer (业务逻辑层)                   │
│  ├─ AgentService (agent 会话生命周期)                     │
│  ├─ MemoryService (三级记忆 CRUD + 向量搜索 + 图谱同步)    │
│  ├─ RAGService (文档索引 + 混合检索 + 问答)               │
│  ├─ ResearchService (深度研究编排)                        │
│  ├─ SkillService (Skill 匹配/生成/评估)                   │
│  ├─ MCPService (工具注册/调用/审批)                       │
│  ├─ GraphContextService (Neo4j 图谱上下文注入)            │
│  ├─ UserGrowthService (动态偏好/用户画像)                 │
│  ├─ ArtifactService (产物文件管理)                        │
│  └─ ScoringService / EvalService / SourceService ...     │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Agent Runtime (agent/runtime/)               │
│  节点执行顺序:                                             │
│  Planner → PermissionGuard → HomeIntent → Router         │
│  → ContextBuilder → SkillMatcher                         │
│  → [Research / RAG / Artifact / Tool / Memory / Skill]   │
│  → Evaluator → FinalResponse                             │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Data Layer                                  │
│  ├─ PostgreSQL (用户/文档/记忆/Agent 运行/Feed/Skill)      │
│  ├─ Qdrant (文档向量 + 记忆向量)                          │
│  ├─ Neo4j (记忆图谱 + 项目知识图谱)                        │
│  └─ Redis (会话/缓存)                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 四、目录结构详解

```
src/web_app/
├── main.py                          # FastAPI 应用入口
├── agent/
│   ├── graph.py                     # 旧版简单节点图（逐步废弃中）
│   ├── state.py                     # AgentState TypedDict
│   ├── schemas.py                   # Pydantic schemas
│   ├── llm/
│   │   ├── factory.py               # LLM 工厂（ChatOpenAI 兼容模式）
│   │   ├── router.py                # 按 Purpose 路由不同模型
│   │   ├── config.py                # LLM 配置获取
│   │   ├── errors.py                # LLM 错误类型
│   │   ├── usage.py                 # LLM 调用记录
│   │   └── embedding.py             # 嵌入模型工厂
│   ├── nodes/                       # 旧版节点（10个节点文件，功能分散）
│   ├── planners/                    # 旧版规划器
│   ├── adapters/                    # MCP 适配器
│   └── runtime/                     # ★ 新版 Runtime（核心）
│       ├── nodes.py                 # RuntimeNodes 类（~2500行，核心节点）
│       ├── state.py                 # AgentRuntimeState
│       ├── planner.py               # 路由规划器
│       ├── router.py                # 意图路由
│       ├── intent_llm.py            # LLM 意图识别
│       ├── intent_schema.py         # 意图 schema
│       ├── checkpoint.py            # 检查点/事件记录
│       ├── checkpointers.py         # 持久化检查点
│       ├── langgraph_status.py      # 状态追踪
│       ├── progress.py              # 进度管理
│       ├── visibility.py            # 可见思考
│       ├── visible_thoughts.py      # 用户可见思考过程
│       └── fallback.py              # 降级策略
├── api/v1/
│   ├── router.py                    # API 路由聚合
│   ├── agent.py                     # Agent 会话 API
│   ├── auth.py                      # 认证 API
│   ├── feed.py                      # Feed 卡片 API
│   ├── documents.py                 # 文档管理 API
│   ├── memory.py                    # 记忆管理 API
│   ├── research.py                  # 深度研究 API
│   ├── skills.py                    # Skill 管理 API
│   ├── artifacts.py                 # Artifact 产物 API
│   ├── approvals.py                 # 审批流 API
│   ├── mcp.py                       # MCP 工具 API
│   ├── health.py                    # 健康检查
│   └── profile.py                   # 用户画像 API
├── services/                        # 业务逻辑层（15+ 服务）
│   ├── agent_service.py             # Agent 运行生命周期
│   ├── memory_service.py            # 三级记忆管理（~700行）
│   ├── rag_service.py               # RAG 检索与问答（~350行）
│   ├── research_service.py          # 深度研究编排
│   ├── skill_service.py             # Skill 匹配与生成
│   ├── mcp_service.py               # MCP 工具管理
│   ├── graph_context_service.py     # Neo4j 图谱上下文
│   ├── user_growth_service.py       # 用户画像动态更新
│   ├── context_service.py           # 上下文构建（上道工序）
│   ├── artifact_service.py          # 产物管理
│   ├── auth_service.py              # 认证
│   ├── profile_service.py           # 用户画像
│   ├── permission_service.py        # 权限检查
│   ├── scoring_service.py           # Feed 评分
│   └── source_service.py            # 数据源
├── context/                         # 上下文构建与质量评估
│   ├── builder.py                   # ContextBuilder（GSSC 四阶段）
│   ├── packets.py                   # ContextPacket / ContextConfig
│   ├── compression.py               # 上下文压缩
│   └── quality.py                   # 上下文质量评估
├── memory/                          # 记忆子系统
│   ├── base.py                      # BaseMemoryStore
│   ├── extractor.py                 # 记忆提取（LLM + 正则双模式）
│   ├── working.py                   # 工作记忆
│   ├── episodic.py                  # 情景记忆
│   ├── semantic.py                  # 语义记忆
│   ├── perceptual.py                # 感知记忆
│   ├── consolidation.py             # 记忆固化
│   └── qdrant_memory_store.py       # Qdrant 记忆向量存储
├── rag/                             # RAG 子系统（8个模块）
│   ├── retriever.py                 # 混合检索器（Parent-Child 架构）
│   ├── embeddings.py                # 嵌入生成
│   ├── vector_store.py              # Qdrant 向量存储
│   ├── bm25.py                      # Python BM25 实现
│   ├── sparse_encoder.py            # 稀疏向量编码
│   ├── query_analyzer.py            # 查询分析器
│   ├── reranker.py                  # 重排序
│   ├── structured_chunker.py        # 结构化分块器
│   ├── chunker.py / chunking.py     # 旧版分块
│   ├── document_loader.py           # 文档加载
│   ├── document_parser.py           # 文档解析
│   └── qdrant_client.py             # Qdrant 客户端
├── research/                        # 深度研究
│   ├── evidence_builder.py          # 证据构建
│   ├── fallback_researcher.py       # 降级研究
│   ├── report_builder.py            # 报告生成
│   └── schemas.py                   # 研究请求/响应 schema
├── graph/                           # ★ Neo4j 图谱层（新增，开发中）
│   ├── __init__.py                  # 空模块标记
│   ├── neo4j_client.py              # Neo4j 连接管理
│   ├── schema.py                    # 图约束定义（18种节点+唯一约束）
│   ├── repositories.py              # 图仓库（CRUD + 上下文查询）
│   ├── memory_projector.py          # 记忆 → 图谱投影
│   └── project_projector.py         # 项目 → 图谱投影
├── db/                              # 数据库层
│   ├── base.py                      # SQLAlchemy Base
│   ├── session.py                   # 会话管理
│   ├── init_db.py                   # 数据库初始化
│   └── repositories/                # 仓库模式（7个 repository）
│       ├── base_repository.py       # 基础 Repository
│       ├── agent_repository.py      # Agent 运行/对话/消息
│       ├── document_repository.py   # 文档/分块
│       ├── memory_repository.py     # 记忆
│       ├── profile_repository.py    # 用户画像
│       ├── research_repository.py   # 研究运行
│       ├── skill_repository.py      # Skill
│       ├── user_repository.py       # 用户
│       ├── feed_repository.py       # Feed 卡片
│       ├── artifact_repository.py   # Artifact
│       └── approval_repository.py   # 审批
├── models/                          # ORM 数据模型
│   ├── orm.py                       # ★ 核心：23个 SQLAlchemy 模型
│   ├── entities.py / memory.py / agent_run.py ...
│   └── schemas/ (common.py)         # 通用 Pydantic schemas
├── feed/                            # Feed 子系统
│   └── dedup.py                     # 去重
├── mcp/                             # MCP 子系统
│   ├── permissions.py               # 权限模型
│   └── registry.py                  # 工具注册表
├── core/                            # 核心配置
│   ├── config.py                    # ★ Settings（200+配置项）
│   ├── constants.py                 # 常量（权限级别等）
│   ├── errors.py                    # 错误类型
│   ├── logging.py                   # 日志
│   └── security.py                  # 安全
├── artifacts/                       # 产物管理
│   ├── generators.py                # 产物生成器
│   └── storage.py                   # 产物存储
├── schemas/                         # 通用 schema
│   └── common.py
└── tests/                           # ★ 测试（大量新增）
    ├── test_rag_*.py                # RAG 测试（~10个文件）
    ├── test_memory_graph.py         # 记忆图谱测试
    ├── test_neo4j_client.py         # Neo4j 客户端测试
    ├── test_graph_context_builder.py # 图谱上下文测试
    ├── test_project_graph.py        # 项目图谱测试
    └── fixtures/                    # 测试数据

scripts/                             # 运维脚本
├── sync_project_graph.py            # 项目图谱同步到 Neo4j
├── ensure_qdrant_indexes.py         # Qdrant 索引维护
├── backfill_memory_vectors.py       # 记忆向量回填
├── run_rag_hybrid_eval.py           # RAG 混合检索评估
└── compare_rag_backends.py          # RAG 后端对比
```

---

## 五、核心模块深入分析

### 5.1 Agent Runtime（`agent/runtime/`）— 心脏

RuntimeNodes（`nodes.py`，~2500行）是整个系统的核心编排器，实现了 11 个关键节点：

| 节点 | 功能 | 调用的服务 |
|------|------|-----------|
| **planner** | 分析用户输入 → 生成 RoutePlan（意图/路由/风险等级） | plan_route() |
| **home_intent_react** | LLM 意图识别（含规则降级） | infer_home_intent_with_llm() |
| **router** | 基本路由判断 | route_user_input() |
| **permission_guard** | 权限等级检测（L0-L4） | PermissionGuard |
| **context_builder** | 构建 GSSC 上下文（记忆/RAG/Feed/图谱/对话历史/动态偏好） | MemoryService, RAGService, GraphContextService, UserGrowthService |
| **skill_matcher** | 匹配已有 Skill | SkillService |
| **research_agent** | 深度研究执行 | ResearchService |
| **rag_agent** | RAG 问答 | RAGService |
| **tool_agent** | MCP 工具调用（含审批暂停/恢复机制） | MCPService, ApprovalRepository |
| **memory_agent** | 显式记忆写入 + 条件提取 | MemoryService |
| **skill_agent** | Skill 复用检测 + 草稿生成 | SkillService |
| **artifact_agent** | Artifact 产物生成 | ArtifactService |
| **final_response** | 流式 LLM 生成最终回答 | LLM Factory |

**路由策略**：
- `chat` — 闲聊，仅 context_builder + final_response
- `rag` / `document_qa` — RAG 知识库问答
- `research` / `feed_research` — 深度研究
- `tool.*` — MCP 工具执行（如 tool.email.send）
- `artifact` — 产物生成
- `skill` — Skill 创建
- `memory` — 显式记忆写入
- `approval` — 进入审批等待（暂停执行）
- `mixed` — 多意图（如 research + rag）

**审批/暂停/恢复机制**：
- Tool Agent 在执行 L3/L4 工具时创建 Approval 记录 → 设置 `status=waiting_approval`
- Graph 中断到 END，前端收到 `waiting_approval` 状态
- 用户 approve/reject 后，通过 resume API 重新进入
- Tool Agent 检测 `resolved_tool_call_ids` 恢复执行

### 5.2 记忆系统（`memory/`）— 三级递进

```
Working Memory (工作记忆) ← 低门槛，不进入长期记忆
  │  importance ≥ 0.7 → 升级
  ▼
Episodic Memory (情景记忆) ← 具体事件/行为记录
  │  importance ≥ 0.8 + evidence ≥ 2 + 匹配语义类别 → 升级
  ▼
Semantic Memory (语义记忆) ← 长期偏好/知识/约束
```

**记忆类别（Semantic Categories）**：
`preference`, `negative_preference`, `project_goal`, `tech_stack`, `boundary`, 
`answer_preference`, `name_preference`, `language_preference`, `tone_preference`, `workflow_pattern`

**搜索策略（三层降级）**：
1. **Qdrant** 语义向量搜索 → 按 65%向量分 + 25%重要性 + 10%时新度排序
2. **PostgreSQL** ILIKE 模糊匹配降级
3. **近期高重要性语义记忆** 兜底

**遗忘策略**（3种）：
- `forget_by_importance` — 低于阈值归档
- `forget_by_time` — 超过最大天数归档
- `forget_by_capacity` — 超出容量限制归档（最低效用的先归档）

**记忆提取**：LLM（qwen3.6-max-preview）主提取 + 正则表达式降级双模式

**记忆同步到 Neo4j**：每次记忆写入/更新/删除后，通过 `memory_projector` 同步到
Neo4j 图谱（UserMemory → Topic/Goal/Preference/Boundary 节点 + 关系）

### 5.3 RAG 系统（`rag/`）— 混合检索

**架构设计**：Parent-Child Chunk 模式
- **Parent Chunk**：较大上下文块（用于 LLM 理解）
- **Child Chunk**：较小索引块（用于向量/BM25 检索）
- 检索时用 Child 块匹配 → 返回对应 Parent 上下文

**混合检索流程**：
```
用户查询 → QueryAnalyzer（分析查询类型）
  → 双路检索：
     ├─ Vector Search（Qdrant dense vector）
     └─ BM25 Search（PostgreSQL 关键词 or Qdrant Cloud BM25）
  → 分数归一化 + 合并去重
  → Reranker 重排序
  → Parent Context 富化
  → 返回 Top-K 结果（含引用元数据）
```

**两种 Hybrid 后端**：
- `python_bm25`：纯 Python BM25 + Qdrant dense vector（默认，最稳定）
- `qdrant_hybrid`：Qdrant Cloud 原生 hybrid search（可选，含自动降级）

**支持的分块策略**：
- Markdown 标题层级分块（H1→H2→H3）
- 表格分块（CSV/Excel 按 sheet）
- Markdown 列表/代码块分块
- 固定大小滑动窗口分块（降级策略）

### 5.4 Neo4j 图谱层（`graph/`）— ★ 当前开发重点

这是当前分支 `feature/bendicaozuo` 正在新增的核心模块。

**两种图谱**：

**A. 记忆图谱（Memory Graph）**
```
(User)-[:HAS_MEMORY]->(UserMemory)-[:MENTIONS]->(MemoryTopic)
(UserMemory)-[:SUPPORTS]->(MemoryGoal)
(UserMemory)-[:SUPPORTS]->(MemoryPreference)
(UserMemory)-[:SUPPORTS]->(MemoryBoundary)
(User)-[:INTERESTED_IN]->(MemoryTopic)
(User)-[:HAS_GOAL]->(MemoryGoal)
(User)-[:PREFERS]->(MemoryPreference)
(User)-[:HAS_BOUNDARY]->(MemoryBoundary)
```

**B. 项目知识图谱（Project Graph）**
```
(Project)-[:HAS_MODULE]->(ProjectModule)
(Project)-[:HAS_SERVICE]->(ProjectService)
(Project)-[:HAS_REPOSITORY]->(ProjectRepository)
(Project)-[:EXPOSES]->(ProjectAPIEndpoint)
(Project)-[:USES_CONFIG]->(ProjectConfigKey)
(Project)-[:USES_TECH]->(ProjectTechnology)
(Project)-[:USES_VECTOR_COLLECTION]->(ProjectQdrantCollection)
(Module/Service/...)-[:PROJECT_RELATION]->(...)
```

**图谱上下文注入**：
`GraphContextService.get_context()` 从 Neo4j 查询相关节点并格式化为上下文文本，
注入到 Agent 的 GSSC 上下文中，用于增强 LLM 对用户长期偏好和项目结构的理解。

**schema.py 定义的约束**（18个唯一约束）：
User, UserMemory, MemoryTopic, MemoryGoal, MemoryPreference, MemoryBoundary,
Project, ProjectModule, ProjectService, ProjectRepository, ProjectAPIEndpoint,
ProjectConfigKey, ProjectQdrantCollection, ProjectTechnology

### 5.5 上下文构建（`context/`）— GSSC 四阶段

`ContextBuilder` 实现了 **GSS+C** 四阶段流水线：

1. **Gather**：收集所有上下文源（profile/memory/evidence/feed/graph_context/conversation_history 等 15 种）
2. **Select**：按路由权重 + 相关性分数排序，在 token budget 内选择
3. **Structure**：按预定义 section 组织（Role & Policies / User Profile / Relevant Memory / Evidence ...）
4. **Compress**：token 超出时智能压缩（丢弃低优先级 section）

**路由自适应权重**：不同路由（chat/feed/research/rag/skill/artifact/tool）对各上下文源的权重不同。
例如 research 路由对 evidence 权重 0.80，chat 路由对 conversation_history 权重 0.95。

**记忆上下文策略**：根据不同 answer_mode 筛选允许注入的记忆类别，防止无关记忆污染上下文。
例如 casual 模式只允许 name/language/tone 偏好，不注入 project_goal/tech_stack。

### 5.6 LLM 模型路由

`agent/llm/router.py` 实现了按 Purpose 分配不同模型的能力：

| Purpose | 优先级 | 默认模型 |
|---------|--------|---------|
| intent | 高 | qwen3.6-max-preview |
| safety | 高 | qwen3.6-max-preview |
| planner | 高 | qwen3.6-max-preview |
| memory | 中 | qwen3.6-max-preview |
| skill | 中 | qwen3.6-max-preview |
| rag | 高 | qwen3.6-max-preview |
| research | 高 | qwen3.7-plus |
| artifact | 中 | qwen3.6-plus |
| final | 中 | qwen3.6-plus |

**LLM 调用模式**：全部通过 `ChatOpenAI`（LangChain）→ 阿里云 DashScope 兼容 API 调用。

### 5.7 数据库模型（23个 ORM 表）

**核心实体**：User, UserProfile, Memory, Document, DocumentChunk, Skill

**Agent 运行**：AgentRun, AgentConversation, AgentChatMessage, AgentStep, AgentEvent

**Feed 生态**：InfoSource, InfoItem, FeedCard, FeedFeedback

**MCP 生态**：MCPServer, MCPTool, ToolCall, Approval

**其他**：Artifact, LLMCall, ResearchRun, EvalRecord

---

## 六、当前分支变更分析（`feature/bendicaozuo`）

当前分支比 main 多了 4 个 commits：

```
201e7d4 完成了rag,下一步neo4j
eb14cf0 下一步，优化时间，qdrant
9bb1ab1 完成了浅度的记忆
08f5874 下一步rdrant
```

**主要变更**（基于 `git diff --stat`）：

| 类别 | 变更 | 说明 |
|------|------|------|
| **新增模块** | `src/web_app/graph/` | Neo4j 图谱层（neo4j_client, schema, repositories, memory_projector, project_projector） |
| **新增服务** | `services/graph_context_service.py` | 图谱上下文注入服务 |
| **新增测试** | 6 个新测试文件 | test_neo4j_client, test_memory_graph, test_project_graph, test_graph_context_builder, test_rag_* |
| **新增脚本** | `scripts/sync_project_graph.py` | 项目图谱同步到 Neo4j |
| **RAG 增强** | bm25.py, sparse_encoder.py, reranker.py, query_analyzer.py, structured_chunker.py | 混合检索完整实现 |
| **配置扩展** | core/config.py | 新增 Neo4j 相关配置 16 项 |
| **记忆增强** | memory_service.py 大幅扩展 | 新增 Qdrant 向量搜索、Neo4j 同步、三种遗忘策略 |
| **上下文增强** | context/builder.py 扩展 | 新增 Graph Context 来源、MEMORY_CONTEXT_POLICY、路由自适应权重 |
| **测试数据** | fixtures/rag_docs/ + fixtures/rag_eval_* | RAG 评测数据集 |
| **依赖新增** | neo4j>=5.28.1, sentence-transformers, tiktoken | Neo4j 驱动 + 嵌入模型 + token 计数 |

**关键：项目图谱同步脚本**（`scripts/sync_project_graph.py`）将本项目的模块/服务/API/配置/技术栈信息同步到 Neo4j，为 Agent 提供项目结构感知能力。

---

## 七、数据流全景

### 7.1 用户对话请求完整流程

```
1. POST /api/v1/agent/chat
   └─ AgentService.run_agent()
      ├─ 创建 AgentRun + AgentChatMessage
      ├─ 构建 AgentRuntimeState
      └─ 执行节点序列（sync → async 混合）:

2. Planner Node
   └─ plan_route() → RoutePlan {intent, route[], risk_level, answer_mode, needs_approval}

3. PermissionGuard Node
   └─ 关键词 + 规则检测 L0-L4 权限等级

4. HomeIntent Node
   └─ LLM 意图识别 (rule fallback) → HomeIntentResult

5. Router Node
   └─ 基本路由确认

6. ContextBuilder Node  ← ★ 最复杂的上下文聚合
   ├─ MemoryService.search_memory() → Qdrant → PG → fallback
   ├─ RAGService.search_evidence() → 向量+BM25 混合检索
   ├─ GraphContextService.get_context() → Neo4j 图谱上下文
   ├─ UserGrowthService → 动态偏好
   ├─ FeedRepository → FeedCard 上下文
   ├─ AgentChatMessageRepository → 对话历史
   └─ ContextBuilder.build() → GSSC 结构化上下文字符串

7. SkillMatcher Node
   └─ SkillService.match_skill() → 语义匹配已有 Skill

8. Domain Agent(s) — 按 RoutePlan.route 执行:
   ├─ research_agent → ResearchService
   ├─ rag_agent → RAGService (search + rerank + LLM answer)
   ├─ tool_agent → MCPService (含审批暂停机制)
   ├─ memory_agent → MemoryService
   ├─ skill_agent → SkillService
   └─ artifact_agent → ArtifactService

9. Evaluator Node
   └─ 评估执行结果，标记状态

10. FinalResponse Node
    ├─ 流式调用 LLM 生成最终回答 → SSE 推送
    ├─ 内存认领守卫（防止 LLM 虚构"已记住"）
    └─ 构建 final_payload（含 answer/artifacts/approval/memory/skill/visible_thoughts/langgraphstatus）

11. 持久化:
    ├─ AgentRun 更新（status/final_answer/elapsed_ms）
    ├─ AgentChatMessage 更新（content + status）
    ├─ AgentStep 记录（每个节点输入输出）
    └─ MemoryService 异步记忆提取 + 写入
```

### 7.2 Feed 刷新流程

```
定时触发 → Feed Refresh
├─ 各源并行抓取（GitHub/ArXiv/Tavily/SerpAPI/DuckDuckGo/RSS）
├─ 去重（content_hash + 相似度）
├─ 与用户画像匹配评分（explicit_related / adjacent_domain / far_domain）
├─ 生成 FeedCard（含 one_sentence_value / why_you / information_gap / suggested_actions）
└─ 存储到 PostgreSQL
```

### 7.3 文档 RAG 索引流程

```
POST /api/v1/documents/upload
├─ 文件保存到 storage/uploads/{user_id}/{doc_id}/
├─ document_parser 解析（支持 PDF/DOCX/MD/TXT/CSV/XLSX）
├─ structured_chunker 结构化分块
│  ├─ Markdown H1/H2/H3 标题层级分块
│  ├─ 表格（CSV/XLSX）分块
│  └─ 固定大小滑动窗口（降级）
├─ embeddings 生成密集向量
├─ Qdrant 向量写入（密集向量 + 可选稀疏向量）
├─ PostgreSQL DocumentChunk 持久化
└─ 返回 ingestion 状态
```

---

## 八、开发历程（基于 git log）

```
当前分支 feature/bendicaozuo:
  201e7d4 完成了rag,下一步neo4j              ← 混合检索完成
  eb14cf0 下一步，优化时间，qdrant              ← Qdrant 优化
  9bb1ab1 完成了浅度的记忆                      ← 记忆系统基础版
  08f5874 下一步rdrant                         ← Qdrant 集成开始

main 分支历史（部分关键节点）:
  422d7ec 下一步，多智能体协作
  85aa336 下一步LLM识别
  4ed9500 完成了邮件的发送，下一步LLM接入识别意图
  31571e3 完成了深度研究，下一步本地电脑操作
  2406e69 完成了信息差卡片基本情况，下一步操作电脑
  4e7d70e 完成了基本搜索展示，下一步从多方搜
  09a8c5f 下一步修远域
  fc60ce1 完成了识图和文件
  815eb33 完成了图片识别，下一步sse输出
  cbb5fa2 完成qrant，下一步rag
  1e2611a 完成了基本记忆完善，下一步是markdown前端渲染
  1d34c14 完成记忆，下一步修复LLM识别回答问题
```

**开发节奏**：项目处于快速迭代阶段，以"完成X，下一步Y"的模式持续推进。当前分支专注于 Neo4j 图谱集成和 RAG 混合检索优化。

---

## 九、设计模式与架构评估

### 9.1 亮点

1. **完整的 Agent OS 闭环**：Feed → Research → Artifact → Memory/Skill 信息处理流水线设计完整
2. **混合检索设计合理**：Dense + Sparse 双路检索 + Reranker + Parent-Child 分块
3. **三级记忆系统**：Working → Episodic → Semantic 递进 + Qdrant 向量搜索 + Neo4j 图谱
4. **路由自适应上下文**：不同任务类型（chat/research/rag/tool）注入不同权重和类别的上下文
5. **LLM 模型路由**：按 Purpose 分配不同能力/成本的模型
6. **审批/暂停/恢复**：支持高风险工具调用的审批流中断与恢复
7. **流式 SSE 输出**：最终回答支持 token 级流式推送到前端

### 9.2 技术债务与风险

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| **agent/nodes/ vs agent/runtime/ 双轨并存** | 中 | 旧 nodes 目录 10 个文件与 runtime 目录功能重叠，应在稳定后清理 |
| **agent/graph.py 旧版入口** | 低 | `run_agent_graph()` 是硬编码的 4 节点序列，与 runtime 的 11 节点编排完全脱节 |
| **RuntimeNodes.nodes.py 过于庞大** | 高 | 单文件 ~2500 行包含所有节点逻辑，应拆分为独立节点文件 |
| **memory_service.py 职责过重** | 中 | ~700 行包含 CRUD + 搜索 + Qdrant 同步 + Neo4j 同步 + 遗忘策略 + 固化，可拆分 |
| **Neo4j 集成仍在早期** | 中 | `graph/__init__.py` 只有一行注释，部分功能未完成 |
| **测试覆盖** | 中 | 大量新增测试但集中于 RAG 和 Neo4j，核心 Runtime 和 Service 测试不足 |
| **配置膨胀** | 低 | Settings 类 ~200 项配置，部分可通过代码常量简化 |

---

## 十、建议与后续方向

### 短期（当前分支收尾）
1. 完成 `graph/` 模块的 `__init__.py`，导出公共 API
2. `sync_project_graph.py` 脚本需要编写项目元数据文件（modules/services/config_keys...）
3. 补充 RuntimeNodes 核心路径的集成测试

### 中期
1. 拆分 `runtime/nodes.py` 为独立节点文件（如 `planner_node.py`, `context_node.py`...）
2. 统一 `agent/nodes/` 和 `agent/runtime/` 目录（删除旧 nodes）
3. 将 `memory_service.py` 拆分为 MemoryCRUD / MemorySearch / MemoryForgetting / MemoryGraphSync
4. 建立 Neo4j 图谱定期同步机制（非一次性脚本）

### 长期
1. 真实 LangGraph 集成（替代自研 Runtime 节点编排）
2. 多用户协作 + 共享知识库
3. 前端独立仓库需要配套更新以支持 Neo4j 图谱可视化
4. 引入 OpenTelemetry 全链路追踪

---

## 附录：关键文件索引

| 用途 | 文件 |
|------|------|
| 应用入口 | `src/web_app/main.py` |
| Agent 编排核心 | `src/web_app/agent/runtime/nodes.py` |
| Agent 状态定义 | `src/web_app/agent/runtime/state.py` |
| LLM 模型路由 | `src/web_app/agent/llm/router.py` |
| 记忆服务 | `src/web_app/services/memory_service.py` |
| RAG 服务 | `src/web_app/services/rag_service.py` |
| RAG 混合检索 | `src/web_app/rag/retriever.py` |
| 上下文构建 | `src/web_app/context/builder.py` |
| 图谱上下文 | `src/web_app/services/graph_context_service.py` |
| Neo4j 客户端 | `src/web_app/graph/neo4j_client.py` |
| 图谱 schema | `src/web_app/graph/schema.py` |
| 图谱仓库 | `src/web_app/graph/repositories.py` |
| 记忆提取 | `src/web_app/memory/extractor.py` |
| 全局配置 | `src/web_app/core/config.py` |
| ORM 模型 | `src/web_app/models/orm.py` |
| API 路由聚合 | `src/web_app/api/v1/router.py` |
| 项目图谱同步 | `scripts/sync_project_graph.py` |
