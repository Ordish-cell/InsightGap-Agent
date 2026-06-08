# Open Deep Research — 信息差 Agent OS 项目全量上下文

> 本文档供 AI 助手快速建立对该项目的完整理解。最后更新: 2026-06-08。

---

## 1. 项目概览

这是一个基于开源项目 [Open Deep Research](https://github.com/langchain-ai/open_deep_research) 进行二次开发的**信息差 Agent OS**。核心目标是构建一个闭环系统：

```
Feed (信息流) → Agent 研究 → Artifact (成果物) → Memory/Skill (记忆沉淀)
```

- **技术栈**: Python 3.10+, FastAPI, LangGraph, LangChain, PostgreSQL, Qdrant, Redis
- **LLM**: 主要使用阿里云 DashScope 的 Qwen 系列模型
- **前端**: Vite + React + TypeScript (独立项目，端口 localhost:5173)
- **作者**: Ordish
- **当前分支**: `feature/backend-step-2`

---

## 2. 目录结构（关键文件）

```
open_deep_research/
├── src/
│   ├── open_deep_research/          # 【上游原始项目 - 不要修改！】
│   │   ├── deep_researcher.py       # LangGraph supervisor-worker 研究主流程
│   │   ├── state.py                 # 状态定义 (AgentState, SupervisorState, ResearcherState)
│   │   ├── configuration.py         # 配置管理
│   │   ├── prompts.py               # LLM 提示词模板
│   │   └── utils.py                 # 工具函数 (搜索工具、token 管理)
│   │
│   ├── web_app/                     # 【二开核心 - 你的 Agent OS】
│   │   ├── main.py                  # FastAPI 入口
│   │   ├── core/
│   │   │   ├── config.py            # 全局配置 (Settings class, 环境变量映射)
│   │   │   ├── constants.py         # 风险等级常量
│   │   │   ├── errors.py            # 自定义异常
│   │   │   ├── logging.py           # 日志配置
│   │   │   └── security.py          # 安全工具
│   │   │
│   │   ├── models/
│   │   │   └── orm.py               # 全部 25+ 个 SQLAlchemy ORM 模型
│   │   │
│   │   ├── db/
│   │   │   ├── base.py              # SQLAlchemy Base
│   │   │   ├── session.py           # DB session 工厂
│   │   │   ├── init_db.py           # 建表脚本
│   │   │   └── repositories/        # 15 个 Repository 类 (数据访问层)
│   │   │
│   │   ├── api/v1/                  # 12 个 API 路由模块
│   │   │   ├── agent.py             # Agent 运行 (SSE 流式/非流式)
│   │   │   ├── feed.py              # Feed 卡片 CRUD
│   │   │   ├── research.py          # 深度研究
│   │   │   ├── memory.py            # 记忆管理
│   │   │   ├── documents.py         # 文档上传 + RAG 搜索
│   │   │   ├── artifacts.py         # 成果物管理
│   │   │   ├── skills.py            # Skill 管理
│   │   │   ├── mcp.py               # MCP 工具
│   │   │   ├── approvals.py         # 审批管理
│   │   │   ├── auth.py              # 认证
│   │   │   └── profile.py           # 用户画像
│   │   │
│   │   ├── services/                # 18 个 Service 类 (业务逻辑层)
│   │   │   ├── agent_service.py     # 【核心】Agent 运行主流程 (994行)
│   │   │   ├── research_service.py   # 深度研究编排
│   │   │   ├── memory_service.py    # 记忆提取/存储/检索
│   │   │   ├── rag_service.py       # RAG 检索问答
│   │   │   ├── document_service.py  # 文档上传/摄入
│   │   │   ├── qwen_multimodal_service.py # 图片分析 (Qwen 多模态)
│   │   │   ├── feed_service.py      # Feed 刷新/混排
│   │   │   ├── skill_service.py     # Skill 匹配/创建
│   │   │   ├── mcp_service.py       # MCP 工具调用
│   │   │   ├── approval_service.py  # 审批流程
│   │   │   ├── auth_service.py      # JWT 认证
│   │   │   ├── profile_service.py   # 用户画像
│   │   │   ├── context_service.py   # 上下文管理
│   │   │   ├── user_growth_service.py # 用户动态画像
│   │   │   ├── permission_service.py# 权限检查
│   │   │   ├── eval_service.py      # 评估
│   │   │   ├── scoring_service.py   # 评分
│   │   │   └── source_service.py    # 数据源管理
│   │   │
│   │   ├── agent/
│   │   │   ├── runtime/             # 【核心】LangGraph 多智能体运行时
│   │   │   │   ├── __init__.py      # 导出 AgentRuntime
│   │   │   │   ├── graph.py         # LangGraph 图定义 (节点+边)
│   │   │   │   ├── nodes.py         # 15 个运行时节点实现 (1800+行)
│   │   │   │   ├── planner.py       # 确定性的关键字路由规划器
│   │   │   │   ├── state.py         # AgentRuntimeState + AgentIntent 枚举
│   │   │   │   ├── events.py        # SSE 事件编码 + channel 定义
│   │   │   │   ├── visible_thoughts.py # 用户可见的进度消息
│   │   │   │   ├── visibility.py    # 可见性检查
│   │   │   │   ├── intent_llm.py    # 基于 LLM 的意图识别
│   │   │   │   ├── intent_schema.py # 意图模式定义
│   │   │   │   ├── fallback.py      # 兜底逻辑
│   │   │   │   ├── checkpoint.py    # 事件记录
│   │   │   │   ├── checkpointers.py # LangGraph checkpoint
│   │   │   │   ├── langgraph_status.py # 执行状态追踪
│   │   │   │   └── progress.py      # 进度管理
│   │   │   ├── llm/                 # LLM 配置层
│   │   │   │   ├── config.py        # LLMSettings (模型名称、keys)
│   │   │   │   ├── router.py        # 按 purpose 路由到具体模型
│   │   │   │   ├── factory.py       # ChatOpenAI 实例工厂
│   │   │   │   ├── embedding.py     # Embedding 模型配置
│   │   │   │   ├── errors.py        # LLM 相关异常
│   │   │   │   └── usage.py         # LLM 调用记录
│   │   │   ├── adapters/            # 外部适配器
│   │   │   ├── nodes/               # 旧版 Agent 节点 (逐步迁移)
│   │   │   └── planners/            # 旧版规划器
│   │   │
│   │   ├── memory/                  # 记忆子系统
│   │   │   ├── base.py              # BaseMemoryStore
│   │   │   ├── working.py           # 工作记忆 (ttl=3600s)
│   │   │   ├── episodic.py          # 情景记忆
│   │   │   ├── semantic.py          # 语义记忆
│   │   │   ├── perceptual.py        # 感知记忆
│   │   │   ├── extractor.py         # 【核心】确定性记忆提取器 (无LLM)
│   │   │   ├── consolidation.py     # 记忆固化 (重要性阈值)
│   │   │   └── qdrant_memory_store.py # Qdrant 向量存储 (语义检索)
│   │   │
│   │   ├── feed/                    # Feed 系统
│   │   │   ├── sources/             # 7 种数据源
│   │   │   │   ├── manager.py       # SearchSourceManager
│   │   │   │   ├── arxiv.py         # Arxiv
│   │   │   │   ├── github.py        # GitHub
│   │   │   │   ├── rss.py           # RSS
│   │   │   │   ├── duckduckgo.py    # DuckDuckGo
│   │   │   │   ├── tavily.py        # Tavily
│   │   │   │   ├── serpapi.py       # SerpApi
│   │   │   │   └── manual_seed.py   # 手动种子数据
│   │   │   ├── mixer.py             # 卡片混排 (30/40/30 比例)
│   │   │   ├── scorer.py            # 评分引擎
│   │   │   ├── normalizer.py        # 数据标准化
│   │   │   ├── dedup.py             # 去重
│   │   │   └── card_generator.py    # 卡片生成
│   │   │
│   │   ├── rag/                     # RAG 系统
│   │   │   ├── vector_store.py      # QdrantVectorStore
│   │   │   ├── embeddings.py        # DashScope / SentenceTransformer 嵌入
│   │   │   ├── chunker.py           # Markdown 分块
│   │   │   └── document_parser.py   # PDF/DOCX/TXT 解析
│   │   │
│   │   ├── mcp/                     # MCP 工具系统
│   │   │   ├── registry.py          # 工具注册
│   │   │   ├── tool_executor.py     # 工具执行
│   │   │   ├── tool_router.py       # 工具路由
│   │   │   ├── local_provider.py    # 本地 Provider
│   │   │   ├── permissions.py       # 权限
│   │   │   ├── schemas.py           # 模式
│   │   │   └── audit.py             # 审计
│   │   │
│   │   ├── context/                 # 上下文系统
│   │   │   ├── builder.py           # 【核心】GSSC 上下文构建器
│   │   │   ├── packets.py           # ContextPacket 数据结构
│   │   │   ├── compression.py       # 压缩
│   │   │   └── quality.py           # 质量评估
│   │   │
│   │   ├── research/                # 研究子系统
│   │   │   ├── open_deep_research_adapter.py # 上游适配器
│   │   │   ├── fallback_researcher.py        # 兜底研究员
│   │   │   ├── report_builder.py             # 报告生成
│   │   │   ├── evidence_builder.py           # 证据构建
│   │   │   └── schemas.py                    # 数据结构
│   │   │
│   │   ├── artifacts/               # 成果物
│   │   │   ├── generators.py        # 生成器
│   │   │   └── storage.py           # 文件存储
│   │   │
│   │   └── schemas/                 # Pydantic 公共模式
│   │
│   ├── security/
│   │   └── auth.py                  # LangGraph 部署认证
│   │
│   └── legacy/                      # 旧版实现 (不再使用)
│       ├── graph.py                  # Plan-and-execute
│       └── multi_agent.py            # Supervisor-researcher
│
├── tests/                           # 测试
├── examples/                        # 示例文档
├── pyproject.toml                   # 项目配置
├── langgraph.json                   # LangGraph 配置
├── .env.example                     # 环境变量模板
└── PROJECT_CONTEXT.md               # 本文档
```

---

## 3. 核心架构：Agent 运行时

### 3.1 请求生命周期

```
HTTP POST /api/v1/agent/runs/stream (SSE 流式)
  └── agent_service.run_agent_async()
        │
        ├── 1. 创建/获取会话 (agent_conversations)
        ├── 2. 创建 AgentRun 记录 (状态: running)
        ├── 3. 加载附件 (图片→多模态分析, 文档→RAG 摄入)
        ├── 4. 判断是否直接图片问答 (fast path, 跳过 Agent 图)
        ├── 5. → AgentRuntime.run() 启动 LangGraph
        └── 6. 持久化结果 + 推送 SSE 事件
```

### 3.2 LangGraph 多智能体图结构

```
START
  → permission_guard      (权限检查)
  → home_intent_react     (LLM 意图识别 + 规则兜底)
  → planner               (确定性路由规划: plan_route)
  → context_builder       (GSSC 上下文组装: 记忆+画像+对话历史+RAG证据)
  → skill_matcher         (Skill 匹配)
  → [conditional dispatch based on route_plan.route]:
      ├── research_agent   (深度研究)
      ├── rag_agent        (知识库检索问答)
      ├── artifact_agent   (成果物生成)
      ├── tool_agent       (MCP 工具调用)
      ├── memory_agent     (记忆写入)
      └── skill_agent      (Skill 检测与创建)
  → evaluator             (质量评估)
  → final_response        (LLM 生成最终回答 + SSE 流式输出)
  → END
```

### 3.3 节点详解

| 节点 | 说明 |
|------|------|
| **permission_guard** | 关键词检测风险等级 (L0-L4), L3+ 需要审批 |
| **home_intent_react** | 先调 LLM 判断意图, 失败则回退到规则引擎 |
| **planner** | 确定性规则引擎 `plan_route()`, 基于 100+ 个中英文关键词匹配, 输出 RoutePlan |
| **context_builder** | 组装 14 种上下文源, 按路由权重选择+排序, 生成结构化 GSSC 文本 |
| **skill_matcher** | 匹配已批准的 Skill, 评分 ≥0.75 自动使用 |
| **research_agent** | 调用 ResearchService 进行深度研究, 生成报告+Artifact+Skill草稿 |
| **rag_agent** | RAG 检索, 从用户知识库中找证据 |
| **artifact_agent** | 将 research/rag 结果保存为 Markdown 文件 |
| **tool_agent** | MCP 工具推断+调用, L3/L4 需审批 |
| **memory_agent** | 提取并保存记忆 (working/episodic/semantic), 显式"记住"最高优先级 |
| **skill_agent** | 评估可复用性, 自动创建 Skill 草稿 |
| **evaluator** | 评估执行结果完整性, 设置最终状态 |
| **final_response** | LLM 流式生成用户可读回答, 内部 JSON 防护 |

### 3.4 路由规划详细逻辑 (planner.py)

12 种意图类型: `chat`, `research`, `feed_research`, `rag`, `artifact`, `tool`, `memory`, `skill`, `mixed`, `tool.email`, `tool.browser`, `tool.comment`, `tool.form_submit`

**优先级**:
1. forced route (payload 中显式指定) → 最高优先
2. LLM 意图识别结果 (home_intent)
3. FeedCard 附带 → `feed_research`
4. 显式记忆写入 ("记住"/"以后") → `memory`, 覆盖 research/rag/artifact
5. 对话回忆检测 → `chat` (不回退到研究)
6. 规则关键词匹配 → 按意图优先级排列
7. 默认 → `chat`

**风险等级**: L0(只读)→L1(搜索)→L2(生成)→L3(外部写入)→L4(删除/支付/转账), L3+ 触发审批

---

## 4. 上下文系统 (ContextBuilder - GSSC)

文件: `src/web_app/context/builder.py`

### 4.1 GSSC 四阶段

1. **Gather**: 收集 14 种来源的数据包, 按路由设定 relevance 权重
2. **Select**: 按权重排序, 在 token 预算内选择 (默认 max_tokens × (1 - reserve_ratio))
3. **Structure**: 分组到 14 个 Section (Role & Policies, User Profile, Task, State, Conversation History, Relevant Memory, Evidence, Information Gap Signals, Tool State, Output Contract, Conversation Summary, Checkpoint Summary, Feed Card Context, Dynamic Preferences)
4. **Compress**: 超预算时截断

### 4.2 上下文来源及权重 (按路由)

| 来源 | chat | research | rag | artifact | tool | skill |
|------|------|----------|-----|----------|------|-------|
| conversation_history | 0.95 | 0.75 | 0.70 | 0.80 | 0.75 | 0.80 |
| memory | 0.75 | 0.65 | 0.55 | 0.65 | 0.45 | 0.80 |
| evidence | 0.35 | 0.80 | 0.90 | 0.60 | 0.70 | 0.40 |
| feed_card | 0.50 | 0.85 | 0.65 | 0.75 | 0.40 | 0.55 |
| dynamic_preferences | 0.65 | 0.60 | - | 0.85 | - | 0.75 |

---

## 5. 记忆系统

### 5.1 三层架构

| 类型 | 存储 | TTL | 说明 |
|------|------|-----|------|
| **working** | PostgreSQL + Qdrant | 3600s | 当前会话上下文, 页面状态 |
| **episodic** | PostgreSQL + Qdrant | 永久 | 具体事件记录 (做了什么, 何时) |
| **semantic** | PostgreSQL + Qdrant | 永久 | 长期偏好/知识 (用户设定, 项目目标, 技术栈) |

### 5.2 记忆提取器 (MemoryExtractor)

文件: `src/web_app/memory/extractor.py` — **无 LLM, 纯正则规则**

提取类型:
- **working**: 当前页面, 选中的 FeedCard
- **episodic**: Skill 匹配/创建, 研究行为, 用户反馈 (正面/负面)
- **semantic**: 项目目标, 技术栈, 边界约束, 产品偏好, 信息兴趣

关键规则:
- 对话性质的闲谈 (问候/感谢/道别) 只保留重要性 ≥0.80 的语义记忆, 且降权至 ≤0.50
- 语义记忆写前去重: Jaccard 相似度 ≥0.55 视为重复

### 5.3 记忆固化

`should_promote`: working→episodic (importance≥0.7), episodic→semantic (importance≥0.8)

### 5.4 记忆检索 (MemoryService.search_memory)

三级回退:
1. **Qdrant 语义搜索** (score≥0.25, 返回 top 8, 按 75% Qdrant score + 25% importance 排序)
2. **PostgreSQL ILIKE** (关键词模糊匹配)
3. **最近重要记忆** (importance≥0.7 的 semantic, top 5)

### 5.5 Qdrant 记忆存储

- 集合: `memory_vectors` (可配置)
- 维度: 384 (与 embedding 模型一致)
- 内容截断: 最多嵌入 4000 字符
- 索引字段: user_id, memory_id, memory_type

---

## 6. Feed 系统

### 6.1 数据流

```
7 种数据源 → 并发抓取 (15s 超时)
  → normalize_raw_item (标准化)
  → deduplicate_items (去重)
  → FeedScorer.score (评分)
  → generate_feed_card (生成卡片)
  → mix_cards (混排 30/40/30)
  → 持久化到 feed_cards 表
```

### 6.2 数据源

| 来源 | 默认状态 | 说明 |
|------|---------|------|
| ManualSeed | 启用 | 手动配置的种子数据, 也是最终 fallback |
| GitHub | 启用 | topics: agent,rag,llm,langgraph,mcp; min_stars≥50 |
| Arxiv | 启用 | 类别: cs.AI, cs.CL, cs.LG |
| DuckDuckGo | 启用 | 中文区域, moderate 安全搜索 |
| RSS | 配置式 | 默认空 URL |
| Tavily | 禁用 | 需 API key |
| SerpApi | 禁用 | 需 API key |

### 6.3 卡片混排

按 relation_type 分成三类, 按 30%/40%/30% 比例混排, 有去重和低置信度限制

---

## 7. RAG 系统

### 7.1 文档摄入流程

```
上传 (PDF/DOCX/TXT/MD/CSV/XLSX/HTML)
  → parse_document (转 Markdown)
  → chunk_markdown (语义分块, ~500 tokens/chunk)
  → embed_texts (384维, DashScope text-embedding-v4)
  → QdrantVectorStore.upsert_chunks
  → 写入 document_chunks 表
```

### 7.2 检索

- `rag_service.search()`: 纯向量检索, 返回 top_k + score
- `rag_service.ask()`: 检索 + 提取式回答 (无 LLM, 拼接前 3 条 chunk)
- `rag_service.search_evidence()`: 轻量证据检索 (供 context_builder 注入 GSSC)

### 7.3 Qdrant 文档存储

- 集合: `agent_os_documents`
- 维度: 384, 距离: Cosine
- 过滤: user_id (必须), document_id (可选)
- 支持 Qdrant Cloud 新版 `query_points` API

---

## 8. LLM 配置体系

### 8.1 模型分层 (按 purpose)

| Purpose | 默认模型 | Tier |
|---------|---------|------|
| intent | qwen3.6-flash | fast |
| safety | qwen3.6-flash | fast |
| memory | qwen3.6-flash | fast |
| skill | qwen3.6-flash | fast |
| planner | qwen3.6-max-preview | balanced |
| rag | qwen3.6-max-preview | balanced |
| artifact | qwen3.6-plus | balanced |
| final | qwen3.6-plus | balanced |
| research | qwen3.7-plus | strong |
| embedding | text-embedding-v4 | fast |

### 8.2 Provider

- `aliyun` (默认): DashScope API, base_url: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `openai_compatible`: 通用 OpenAI 兼容 API
- `disabled`: 完全关闭 LLM

### 8.3 多模态 (图片)

- 模型: `qwen_vision_model` (默认 qwen3.6-plus, 支持 qwen-vl 系列)
- 限制: 单图 ≤10MB, 每次最多 5 张, 总 base64 ≤40MB
- 两种模式:
  - `analyze_images()`: 返回结构化内部上下文 (供 RAG/attachment_context)
  - `answer_image_question()`: 返回用户可读的自然语言回答

---

## 9. 图片识别系统 (最新完成)

文件: `src/web_app/services/qwen_multimodal_service.py`, `src/web_app/services/agent_service.py`

### 9.1 触发条件

在 `agent_service.py` 的 `run_agent_async()` 中, 最先检查:

```python
_is_direct_image_question(user_input, attachments)
```

判断标准 (优先级):
1. 附件中有 kind=image 的文件
2. 用户消息包含图片分析关键词 (中英文 20+ 个: "分析图片", "analyze this image" 等)
3. 短消息 (≤30字) + 有图片 → 自动视为图片问题
4. 但如果同时有文档附件且用户提到了文档关键词 → 走正常 Agent 流程

### 9.2 执行路径

如果判定为直接图片问题 → **快速通道**:
1. 调用 `qwen_multimodal_service.answer_image_question()` 获取自然语言回答
2. 直接持久化 assistant_message (标记 `direct_image_answer: true`)
3. 直接推送 SSE 事件 (answer_started → answer_delta → answer_completed)
4. 跳过整个 Agent Runtime LangGraph

如果非直接图片问题 (有文档+图片混合) → 正常 Agent 流程, 图片分析结果注入 attachment_context

### 9.3 清理输出

`_clean_direct_image_answer()` 移除以 `[Image Understanding]`, `Image:`, `Description:`, `OCR:` 等开头的内部标签行

---

## 10. SSE 事件系统

文件: `src/web_app/agent/runtime/events.py`

### 10.1 事件类型与 Channels

| 事件 | 可见性 | Channel | 说明 |
|------|--------|---------|------|
| visible_thought_delta | user | thinking | Agent 思考进度 |
| visible_progress_delta | user | thinking | 进度更新 |
| answer_started | user | answer | 回答开始 |
| answer_delta | user | answer | 流式回答片段 |
| answer_completed | user | answer | 回答完成 |
| tool_call_started/delta/completed/failed | user | tool | 工具调用 |
| approval_required/granted/rejected | user | status | 审批 |
| run_created/completed/failed/paused/resumed | user | status | 运行状态 |
| milestone_started/completed | user | thinking | 里程碑 |

### 10.2 SSE 格式

```
event: {event_type}
data: {json_string}

```

前端根据 `display_channel` 决定渲染到哪个 UI 区域 (thinking/answer/tool/status)

---

## 11. 数据库架构 (PostgreSQL)

### 11.1 核心表 (15个)

| 表名 | 说明 | 关键字段 |
|------|------|---------|
| users | 用户 | email, hashed_password, nickname, status |
| user_profiles | 用户画像 | segment, goals, interests, feed_ratio_config |
| memories | 记忆 | user_id, memory_type(working/episodic/semantic), content, importance, qdrant_point_id |
| feed_cards | Feed卡片 | user_id, info_item_id, title, one_sentence_value, information_gap, final_score, exposure_bucket |
| info_items | 信息条目 | title, content, source_url, content_hash (去重) |
| documents | 文档 | user_id, filename, file_path, file_type, status, metadata_json |
| document_chunks | 文档分块 | document_id, chunk_index, content, qdrant_point_id |
| agent_runs | Agent运行 | user_id, conversation_id, thread_id, status, final_answer, langgraphstatus_json |
| agent_conversations | 会话 | user_id, conversation_id(uuid), title, status, thread_id |
| agent_chat_messages | 聊天消息 | conversation_id, role(user/assistant), content, status |
| agent_steps | 步骤记录 | run_id, node_name, input, output, status |
| agent_events | 事件记录 | run_id, event_type, node_name, payload_json |
| llm_calls | LLM调用日志 | run_id, purpose, provider, model, latency_ms, input_tokens |
| research_runs | 研究运行 | id(uuid), feed_card_id, query, status, findings, markdown_report |
| skills | 技能 | user_id, name, trigger_text, tool_plan, safety_level, status(draft/approved/disabled) |

### 11.2 辅助表

| 表名 | 说明 |
|------|------|
| approvals | 审批记录 |
| artifacts | 成果物 |
| tool_calls | MCP工具调用 |
| mcp_servers / mcp_tools | MCP 配置 |
| info_sources | 信息来源配置 |
| feed_feedback | 用户对 Feed 的反馈 |
| eval_records | 评估记录 |

---

## 12. 关键设计模式

### 12.1 Repository 模式
所有数据库操作通过 `db/repositories/` 下的类封装, 每个模型对应一个 Repository

### 12.2 Service 模式
业务逻辑在 `services/` 下, 每个 Service 是单例 (`xxx_service = XxxService()`)

### 12.3 降级策略 (多层次 fallback)
- **意图识别**: LLM → 规则引擎
- **记忆检索**: Qdrant → PostgreSQL ILIKE → 最近重要记忆
- **研究执行**: 上游 Open Deep Research → FallbackResearcher
- **最终回答**: LLM → 规则生成的兜底文案
- **Feed**: 真实源 → ManualSeed 种子数据

### 12.4 非阻塞设计
- RAG 搜索失败不阻断 Agent 流水线
- 记忆提取失败不阻断 Agent Run
- 图片分析失败返回友好错误信息

---

## 13. API 端点总览

| 前缀 | 用途 | 关键端点 |
|------|------|---------|
| /api/v1/agent | Agent 运行 | POST /runs/stream (SSE), CRUD conversations |
| /api/v1/feed | Feed | GET /home, POST /refresh, POST /cards/:id/research |
| /api/v1/research | 研究 | POST /runs, POST /deep |
| /api/v1/documents + /rag | RAG | POST /upload, POST /ingest, POST /rag/search, POST /rag/ask |
| /api/v1/memory | 记忆 | POST /add, POST /search, POST /consolidate, POST /forget, GET /growth-profile |
| /api/v1/artifacts | 成果物 | CRUD |
| /api/v1/skills | 技能 | CRUD + approve/disable |
| /api/v1/mcp | MCP | 工具列表 + 执行 |
| /api/v1/approvals | 审批 | approve/reject |
| /api/v1/auth | 认证 | login/register |
| /api/v1/profile | 画像 | get/update |
| /api/health | 健康检查 | GET |

---

## 14. 提交历史 (最近 10 次)

```
815eb33 ← 当前 HEAD: 完成图片识别，下一步 SSE 输出
cbb5fa2: 完成 Qdrant 集成，下一步 RAG
1e2611a: 完成基本记忆，下一步 Markdown 前端渲染
1d34c14: 完成记忆，下一步修复 LLM 识别回答
d99bd42: 下一步学 Codex 风格
a3f3917: Bug 修复
40007dd: 完成基本对话，下一步优化流式输出
c84bdde: 完成对话基本功能，输出和思考过程还有问题
49c5274: 完成会话管理，下一步正常对话
c3b38e5: 完成基本搭建
eff3b0b: 完成信息接入
7b4b043: 完成基本 Agent 流转
```

---

## 15. 当前状态总结

### 已完成
- 基础 Agent 框架 (LangGraph 多智能体)
- 会话管理 (CRUD, 多轮对话)
- Feed 系统 (7 源聚合, 评分, 混排)
- 记忆系统 (三层 + Qdrant 语义检索 + 确定性提取器)
- RAG 系统 (文档上传/摄入/检索)
- 图片识别 (Qwen 多模态, 直接问答快速通道)
- 流式输出 (SSE)
- Skill 系统 (匹配/创建/审批)

### 已知待优化
- SSE 流式输出体验优化 (下一步)
- Markdown 前端渲染
- LLM 回答质量 (有时输出 JSON 而非自然语言, 已有防护但需改进)
- Research Adapter 主要是 mock, 真实调用上游需配置
- Codex 风格的界面布局

### 用户约束 (从记忆中提取)
- 不要修改 `src/open_deep_research/` 目录
- 不要引入 Exa、Neo4j、真实电脑操作
- 不重写 Agent Runtime, 不破坏现有 API
- 偏好中文表达, 简洁, 产品化风格
- 不要把内部 JSON 暴露给普通用户

---

## 16. 开发命令

```bash
# 环境变量
cp .env.example .env
# 编辑 .env 配置数据库、LLM API key 等

# 启动开发服务器
uvx langgraph dev                    # LangGraph Studio (上游项目)
uvicorn src.web_app.main:app --reload # FastAPI (Agent OS)

# 代码质量
ruff check
mypy

# 测试
python tests/run_evaluate.py
```
