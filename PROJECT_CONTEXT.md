# Open Deep Research -- 信息差 Agent OS 项目全量上下文

> 本文档供 AI 助手快速建立对该项目的完整理解。最后更新: 2026-06-08。

---

## 1. 项目概览

本项目基于开源项目 [Open Deep Research](https://github.com/langchain-ai/open_deep_research) 进行二次开发，构建了一个**信息差 Agent OS**。核心闭环：

```
Feed (信息流接入) → Agent 研究 → Artifact (成果物) → Memory/Skill (记忆沉淀)
```

- **技术栈**: Python 3.10+, FastAPI, LangGraph, LangChain, PostgreSQL, Qdrant, Redis
- **LLM**: 主要使用阿里云 DashScope (百炼) 的 Qwen 系列模型
- **前端**: Vite + React + TypeScript (独立项目，端口 localhost:5173)
- **作者**: Ordish
- **当前分支**: `feature/bendicaozuo`
- **上游来源**: https://github.com/langchain-ai/open_deep_research

---

## 2. 目录结构（关键文件）

```
open_deep_research/
├── src/
│   ├── open_deep_research/          # 【上游开源项目 - 不要修改！】
│   │   ├── deep_researcher.py       # ★ LangGraph supervisor-worker 研究主流程 (719行)
│   │   │   │                        #   节点: clarify_with_user → write_research_brief
│   │   │   │                        #        → supervisor子图 → final_report_generation
│   │   │   │                        #   Supervisor委托Research, Researcher子图并行执行
│   │   │   ├── state.py             # AgentState, SupervisorState, ResearcherState + 结构化输出
│   │   │   ├── configuration.py     # Configuration (Pydantic) — 所有可配置参数
│   │   │   ├── prompts.py           # 全部LLM提示词模板 (9个prompt)
│   │   │   └── utils.py             # 工具函数: Tavily搜索, MCP, think_tool, token管理 (926行)
│   │   │
│   │   ├── web_app/                 # 【二开核心 - Agent OS】
│   │   │   ├── main.py              # FastAPI 入口 (title="Open Deep Research Agent OS API")
│   │   │   ├── core/
│   │   │   │   ├── config.py        # Settings (pydantic-settings, 100+配置项, 从.env加载)
│   │   │   │   ├── constants.py     # 风险等级常量 (L0-L4)
│   │   │   │   ├── errors.py        # 自定义异常类
│   │   │   │   ├── logging.py       # 日志配置
│   │   │   │   └── security.py      # 安全工具函数
│   │   │   │
│   │   │   ├── models/              # SQLAlchemy ORM 模型 (25+ 表)
│   │   │   │   ├── orm.py           # 主ORM定义 (未在Glob中直接出现，entities.py等拆分)
│   │   │   │   ├── agent_run.py     # AgentRun模型
│   │   │   │   ├── approval.py      # Approval模型
│   │   │   │   ├── artifact.py      # Artifact模型
│   │   │   │   ├── document.py      # Document/DocumentChunk模型
│   │   │   │   ├── entities.py      # User实体
│   │   │   │   ├── feed_card.py     # FeedCard/InfoItem模型
│   │   │   │   └── eval_record.py   # EvalRecord模型
│   │   │   │
│   │   │   ├── db/
│   │   │   │   ├── base.py          # SQLAlchemy Base声明
│   │   │   │   ├── session.py       # DB session工厂 (SessionLocal)
│   │   │   │   ├── init_db.py       # 建表脚本
│   │   │   │   └── repositories/    # 11+ Repository (数据访问层)
│   │   │   │       ├── base_repository.py
│   │   │   │       ├── agent_repository.py
│   │   │   │       ├── approval_repository.py
│   │   │   │       ├── artifact_repository.py
│   │   │   │       ├── document_repository.py
│   │   │   │       ├── mcp_repository.py
│   │   │   │       ├── memory_repository.py
│   │   │   │       ├── profile_repository.py
│   │   │   │       ├── research_repository.py
│   │   │   │       ├── skill_repository.py
│   │   │   │       └── user_repository.py
│   │   │   │
│   │   │   ├── api/v1/              # 12 个 API 路由模块
│   │   │   │   ├── router.py        # APIRouter汇总
│   │   │   │   ├── agent.py         # Agent运行 (SSE流式/非流式)
│   │   │   │   ├── research.py      # 深度研究 (异步后台任务)
│   │   │   │   ├── memory.py        # 记忆增删查改+固化
│   │   │   │   ├── documents.py     # 文档上传+RAG搜索
│   │   │   │   ├── artifacts.py     # 成果物管理
│   │   │   │   ├── skills.py        # Skill管理
│   │   │   │   ├── mcp.py           # MCP工具
│   │   │   │   ├── approvals.py     # 审批管理
│   │   │   │   ├── auth.py          # 认证 (login/register)
│   │   │   │   ├── profile.py       # 用户画像
│   │   │   │   └── health.py        # 健康检查 /api/health
│   │   │   │
│   │   │   ├── services/            # 业务逻辑层 (单例模式)
│   │   │   │   ├── agent_service.py     # 【核心】Agent运行主流程
│   │   │   │   ├── research_service.py  # 深度研究编排 (adapter/fallback)
│   │   │   │   ├── memory_service.py    # 记忆提取+存储+检索 (三级回退)
│   │   │   │   ├── rag_service.py       # RAG检索问答
│   │   │   │   ├── document_service.py  # 文档上传/解析/摄入
│   │   │   │   ├── qwen_multimodal_service.py # 图片分析 (Qwen多模态)
│   │   │   │   ├── feed_service.py      # Feed刷新/混排
│   │   │   │   ├── skill_service.py     # Skill匹配/创建/审批
│   │   │   │   ├── mcp_service.py       # MCP工具调用
│   │   │   │   ├── approval_service.py  # 审批流程
│   │   │   │   ├── auth_service.py      # JWT认证
│   │   │   │   ├── profile_service.py   # 用户画像
│   │   │   │   ├── context_service.py   # 上下文组装
│   │   │   │   ├── user_growth_service.py  # 用户动态画像/成长
│   │   │   │   ├── permission_service.py   # 权限检查
│   │   │   │   ├── eval_service.py      # 质量评估
│   │   │   │   ├── scoring_service.py   # Feed评分
│   │   │   │   └── source_service.py    # Feed数据源管理
│   │   │   │
│   │   │   ├── agent/                # Agent运行时 (LangGraph多智能体)
│   │   │   │   ├── state.py          # AgentState定义
│   │   │   │   ├── schemas.py        # Pydantic数据结构
│   │   │   │   ├── graph.py          # 备用图定义
│   │   │   │   ├── runtime/          # 【核心运行时】
│   │   │   │   │   ├── __init__.py   # 导出AgentRuntime
│   │   │   │   │   ├── graph.py      # ★ LangGraph图定义 (AgentRuntime._build_langgraph)
│   │   │   │   │   ├── nodes.py      # ★ RuntimeNodes类 (15个节点方法)
│   │   │   │   │   ├── router.py     # 条件路由 dispatch_next_route_node
│   │   │   │   │   ├── planner.py    # 确定性关键字路由 plan_route()
│   │   │   │   │   ├── state.py      # AgentRuntimeState TypedDict + AgentIntent枚举
│   │   │   │   │   ├── events.py     # SSE事件类型+display_channel定义
│   │   │   │   │   ├── visible_thoughts.py # 用户可见思考进度
│   │   │   │   │   ├── visibility.py # 可见性检查
│   │   │   │   │   ├── intent_llm.py # 基于LLM的意图识别
│   │   │   │   │   ├── intent_schema.py  # 意图模式定义
│   │   │   │   │   ├── fallback.py   # 兜底逻辑 (无LangGraph时)
│   │   │   │   │   ├── checkpoint.py # 事件记录
│   │   │   │   │   ├── checkpointers.py # LangGraph checkpoint
│   │   │   │   │   ├── langgraph_status.py # 执行状态追踪 (max_steps=12)
│   │   │   │   │   └── progress.py   # 进度管理
│   │   │   │   ├── llm/              # LLM配置层
│   │   │   │   │   ├── config.py     # LLMSettings (model_name, keys, timeout等)
│   │   │   │   │   ├── router.py     # 按purpose路由到具体模型 (intent→fast, research→strong等)
│   │   │   │   │   ├── factory.py    # ChatOpenAI兼容实例工厂
│   │   │   │   │   ├── embedding.py  # Embedding模型配置 (DashScope/OpenAI)
│   │   │   │   │   ├── errors.py     # LLM相关异常
│   │   │   │   │   └── usage.py      # LLM调用日志记录
│   │   │   │   ├── adapters/         # 外部适配器
│   │   │   │   │   ├── mcp_adapter.py
│   │   │   │   │   └── open_deep_research_adapter.py  # 上游ODR适配
│   │   │   │   ├── nodes/            # 备用Agent节点实现
│   │   │   │   └── planners/         # 旧版规划器
│   │   │   │
│   │   │   ├── memory/               # 记忆子系统
│   │   │   │   ├── base.py           # BaseMemoryStore抽象类
│   │   │   │   ├── working.py        # 工作记忆 (Redis TTL=3600s)
│   │   │   │   ├── episodic.py       # 情景记忆 (具体事件)
│   │   │   │   ├── semantic.py       # 语义记忆 (长期偏好/知识)
│   │   │   │   ├── perceptual.py     # 感知记忆 (原始感知数据)
│   │   │   │   ├── extractor.py      # ★ 确定性记忆提取器 (纯正则规则, 无LLM)
│   │   │   │   ├── consolidation.py  # 记忆固化 (重要性阈值判断)
│   │   │   │   └── qdrant_memory_store.py # Qdrant向量存储 (memory_vectors集合, 384维)
│   │   │   │
│   │   │   ├── feed/                 # Feed系统
│   │   │   │   ├── sources/          # 7种数据源
│   │   │   │   │   ├── manager.py    # SearchSourceManager (统一调度)
│   │   │   │   │   ├── arxiv.py      # Arxiv论文
│   │   │   │   │   ├── github.py     # GitHub仓库 (min_stars≥50)
│   │   │   │   │   ├── rss.py        # RSS订阅
│   │   │   │   │   ├── duckduckgo.py # DuckDuckGo搜索 (中文区域)
│   │   │   │   │   ├── tavily.py     # Tavily搜索
│   │   │   │   │   ├── serpapi.py    # SerpApi搜索
│   │   │   │   │   └── manual_seed.py # 手动种子数据 (final fallback)
│   │   │   │   ├── scorer.py         # FeedScorer 评分引擎
│   │   │   │   ├── normalizer.py     # 数据标准化
│   │   │   │   ├── dedup.py          # 去重逻辑
│   │   │   │   ├── mixer.py          # 卡片混排 (30/40/30比例)
│   │   │   │   └── card_generator.py # 卡片生成
│   │   │   │
│   │   │   ├── rag/                  # RAG系统
│   │   │   │   ├── vector_store.py   # QdrantVectorStore (Qdrant Cloud API)
│   │   │   │   ├── embeddings.py     # 嵌入生成 (DashScope/SentenceTransformer)
│   │   │   │   ├── chunker.py        # Markdown语义分块 (~500 tokens/chunk)
│   │   │   │   └── document_parser.py # PDF/DOCX/TXT/CSV/XLSX/HTML解析
│   │   │   │
│   │   │   ├── mcp/                  # MCP工具系统
│   │   │   │   ├── registry.py       # 工具注册中心
│   │   │   │   ├── tool_executor.py  # 工具执行 (含审批检查)
│   │   │   │   ├── tool_router.py    # 工具路由 (匹配工具→执行)
│   │   │   │   ├── local_provider.py # 本地工具Provider
│   │   │   │   ├── permissions.py    # 权限控制
│   │   │   │   ├── schemas.py        # 数据结构
│   │   │   │   └── audit.py          # 审计日志
│   │   │   │
│   │   │   ├── context/              # 上下文系统 (GSSC)
│   │   │   │   ├── builder.py        # ★ ContextBuilder — G/S/S/C四阶段
│   │   │   │   ├── packets.py        # ContextPacket数据结构
│   │   │   │   ├── compression.py    # 上下文压缩
│   │   │   │   └── quality.py        # 质量评估
│   │   │   │
│   │   │   ├── research/             # 研究子系统
│   │   │   │   ├── open_deep_research_adapter.py # 上游ODR适配器
│   │   │   │   ├── fallback_researcher.py        # 兜底研究员
│   │   │   │   ├── report_builder.py             # 报告生成
│   │   │   │   ├── evidence_builder.py           # 证据构建
│   │   │   │   └── schemas.py                    # 数据结构
│   │   │   │
│   │   │   ├── artifacts/           # 成果物
│   │   │   │   ├── generators.py    # ArtifactGenerator
│   │   │   │   └── storage.py       # 文件存储 (本地磁盘)
│   │   │   │
│   │   │   └── schemas/             # Pydantic公共模式
│   │   │
│   │   ├── security/
│   │   │   └── auth.py              # LangGraph部署认证 (Supabase JWT)
│   │   │
│   │   └── legacy/                  # 旧版实现 (不再活跃开发)
│   │       ├── graph.py             # Plan-and-execute + human-in-the-loop
│   │       ├── multi_agent.py        # Supervisor-researcher多Agent
│   │       ├── configuration.py
│   │       ├── state.py
│   │       ├── prompts.py
│   │       ├── utils.py
│   │       └── tests/
│   │
│   ├── tests/                       # 评估框架
│   │   ├── run_evaluate.py          # LangSmith评估主脚本
│   │   ├── evaluators.py            # 6个评估器 (overall/relevance/structure/correctness/groundedness/completeness)
│   │   ├── prompts.py               # 评估提示词
│   │   ├── pairwise_evaluation.py   # 配对比较评估
│   │   ├── supervisor_parallel_evaluation.py  # 多线程并行评估
│   │   ├── extract_langsmith_data.py # LangSmith数据导出
│   │   └── expt_results/            # 实验结果 (JSONL)
│   │
│   ├── examples/                    # 研究示例
│   │   ├── arxiv.md
│   │   ├── pubmed.md
│   │   └── inference-market.md
│   │
│   ├── .github/
│   │   ├── workflows/claude.yml            # Claude Code Bot (@claude触发)
│   │   ├── workflows/claude-code-review.yml # 自动PR Code Review
│   │   └── dependabot.yml                  # 每周pip+actions依赖更新
│   │
│   ├── langgraph.json               # LangGraph部署配置 (入口: deep_researcher)
│   ├── pyproject.toml               # 项目依赖 (60+包) + ruff配置
│   ├── .env.example                 # 环境变量模板 (100+项)
│   ├── CLAUDE.md                    # Claude Code项目指令
│   └── PROJECT_CONTEXT.md           # 本文档
```

---

## 3. 核心架构：Agent 运行时

### 3.1 请求生命周期

```
HTTP POST /api/v1/agent/runs/stream (SSE流式)
  └── agent_service.run_agent_async()
        │
        ├── 1. 创建/获取会话 (agent_conversations表)
        ├── 2. 创建AgentRun记录 (status: running)
        ├── 3. 处理附件 (图片→多模态分析, 文档→RAG摄入)
        ├── 4. 判断是否直接图片问答 ★ (fast path, 跳过LangGraph)
        ├── 5. → AgentRuntime.run() 启动LangGraph多智能体图
        └── 6. 持久化结果 + 推送SSE events (agent_events表)
```

### 3.2 LangGraph 多智能体图结构

```
START
  → permission_guard       (权限检查, L0-L4风险等级)
  → home_intent_react      (LLM意图识别 + 规则引擎兜底)
  → planner                (确定性路由规划: plan_route())
  → context_builder        (GSSC上下文组装: 记忆+画像+对话+RAG证据+Feed)
  → skill_matcher          (Skill自动匹配, score≥0.75自动使用)
  → [条件路由 dispatch_next_route_node]:
      ├── research_agent   (深度研究: 调用ResearchService)
      ├── rag_agent        (知识库检索问答: RAGService)
      ├── artifact_agent   (成果物生成保存)
      ├── tool_agent       (MCP工具推断+调用, L3/L4需审批)
      ├── memory_agent     (记忆提取+保存, 三层记忆)
      └── skill_agent      (Skill检测+自动创建草稿)
  → evaluator              (质量评估, 完整性检查)
  → final_response         (LLM流式生成最终回答, 内部JSON防护)
  → END
```

多路由支持: route_plan.route 是列表，agent按序执行，每个agent完成后router决定下一步(下一个agent/评估/最终回答)

### 3.3 14个运行时节点详解

| 节点 | 对应方法 | 核心逻辑 |
|------|---------|---------|
| **permission_guard** | `nodes.permission_guard()` | 关键词检测风险等级(L0-L4), L3+触发审批, blocked→final_response |
| **home_intent_react** | `nodes.home_intent_react()` | LLM意图分类 + 规则引擎兜底, 输出AgentIntent枚举 |
| **planner** | `nodes.planner()` | `plan_route()`: 基于100+中英文关键词+强制路由+LLM意图+FeedCard, 输出RoutePlan |
| **context_builder** | `nodes.context_builder()` | GSSC四阶段: 收集14种来源→选择(按路由权重)→结构化(14个Section)→压缩 |
| **skill_matcher** | `nodes.skill_matcher()` | 检索已批准Skill, 向量+关键词匹配, score≥0.75自动触发 |
| **research_agent** | `nodes.research_agent()` | 调用ResearchService, 搜索+分析+证据构建+Report+Skill草稿 |
| **rag_agent** | `nodes.rag_agent()` | RAG检索用户知识库, 提取证据 |
| **artifact_agent** | `nodes.artifact_agent()` | 将research/rag结果保存为Markdown成果物 |
| **tool_agent** | `nodes.tool_agent()` | MCP工具名推断(ToolRouter)→工具执行(L3/L4需审批)→结果注入上下文 |
| **memory_agent** | `nodes.memory_agent()` | MemoryExtractor提取→去重→保存(working/episodic/semantic)→同步Qdrant |
| **skill_agent** | `nodes.skill_agent()` | 评估可复用性, 自动创建Skill草稿(status=draft) |
| **skill_draft_detector** | `nodes.skill_draft_detector()` | 检测是否需要创建Skill草稿 |
| **evaluator** | `nodes.evaluator()` | 评估执行结果完整性, 设置run最终状态 |
| **final_response** | `nodes.final_response()` | LLM流式生成用户可读回答, 防护内部JSON泄露, 推送SSE answer_delta |

### 3.4 路由规划详细逻辑 (planner.py: plan_route())

**12种意图类型**: `chat`, `research`, `feed_research`, `rag`, `artifact`, `tool`, `memory`, `skill`, `mixed`, `tool.email`, `tool.browser`, `tool.comment`, `tool.form_submit`

**优先级体系**:
1. forced_route (payload中显式指定) → 最高优先
2. LLM意图识别结果 (home_intent_react)
3. FeedCard附带 → `feed_research`
4. 显式记忆写入关键词 ("记住"/"以后"/"别忘了") → `memory`, 覆盖 research/rag/artifact
5. 对话回忆检测 ("之前"/"上次"/"说过") → `chat` (不回退到研究)
6. 规则关键词匹配 (100+中英文关键词) → 按意图优先级排列
7. 默认 → `chat`

**风险等级**:
| 等级 | 说明 | 示例 | 行为 |
|------|------|------|------|
| L0 | 只读 | 对话, 搜索, RAG | 直接放行 |
| L1 | 搜索 | 搜索, Feed | 直接放行 |
| L2 | 生成 | 研究, 生成Artifact | 直接放行 |
| L3 | 外部写入 | 发送邮件, 评论 | 需要审批 |
| L4 | 破坏性 | 删除, 支付, 转账 | 需要审批 |

---

## 4. 上下文系统 (ContextBuilder — GSSC)

文件: `src/web_app/context/builder.py`

### 4.1 GSSC 四阶段

1. **Gather**: 收集14种来源的ContextPacket, 按路由设定relevance权重
2. **Select**: 按权重排序, 在token预算内选择 (max_tokens × (1 - reserve_ratio))
3. **Structure**: 分组到14个Section (Role & Policies, User Profile, Task, State, Conversation History, Relevant Memory, Evidence, Information Gap Signals, Tool State, Output Contract, Conversation Summary, Checkpoint Summary, Feed Card Context, Dynamic Preferences)
4. **Compress**: 超预算时截断低权重来源

### 4.2 上下文来源及权重 (按路由)

| 来源 | chat | research | rag | artifact | tool | skill |
|------|------|----------|-----|----------|------|-------|
| conversation_history | 0.95 | 0.75 | 0.70 | 0.80 | 0.75 | 0.80 |
| memory (semantic+episodic) | 0.75 | 0.65 | 0.55 | 0.65 | 0.45 | 0.80 |
| evidence (RAG结果) | 0.35 | 0.80 | 0.90 | 0.60 | 0.70 | 0.40 |
| feed_card (信息差卡片) | 0.50 | 0.85 | 0.65 | 0.75 | 0.40 | 0.55 |
| dynamic_preferences | 0.65 | 0.60 | - | 0.85 | - | 0.75 |

---

## 5. 记忆系统

### 5.1 三层架构

| 类型 | 存储 | TTL | 说明 |
|------|------|-----|------|
| **working** | Redis + PostgreSQL + Qdrant | 3600s | 当前会话上下文, 页面状态, 选中卡片 |
| **episodic** | PostgreSQL + Qdrant | 永久 | 具体事件记录 (做了什么, 何时, 结果) |
| **semantic** | PostgreSQL + Qdrant | 永久 | 长期偏好/知识 (用户设定, 项目目标, 技术栈, 兴趣) |
| **perceptual** | - | - | 原始感知数据缓存 |

### 5.2 记忆提取器 (MemoryExtractor) ★

文件: `src/web_app/memory/extractor.py` — **无LLM, 纯正则规则引擎**

提取类型:
- **working**: 当前页面, 选中的FeedCard ID
- **episodic**: Skill匹配/创建, 研究行为, 用户反馈 (正面/负面)
- **semantic**: 项目目标, 技术栈, 边界约束, 产品偏好, 信息兴趣

关键规则:
- 对话性质的闲谈 (问候/感谢/道别) 只保留 importance≥0.80 的语义记忆, 且降权至 ≤0.50
- 语义记忆写前去重: Jaccard相似度 ≥0.55 视为重复, 合并+更新importance

### 5.3 记忆固化 (consolidation.py)

`should_promote()`:
- working → episodic: importance ≥ 0.7
- episodic → semantic: importance ≥ 0.8

### 5.4 记忆检索 (MemoryService.search_memory)

三级回退策略:
1. **Qdrant 语义搜索** — score≥0.25, 返回top 8, 按 `0.75 × Qdrant score + 0.25 × importance` 排序
2. **PostgreSQL ILIKE** — 关键词模糊匹配 (fallback)
3. **最近重要记忆** — importance≥0.7的semantic记忆, top 5 (final fallback)

### 5.5 Qdrant 记忆存储 (qdrant_memory_store.py)

- 集合: `memory_vectors` (可配置: `MEMORY_QDRANT_COLLECTION`)
- 维度: 384 (与embedding模型一致, text-embedding-v4)
- 内容截断: 最多嵌入前4000字符
- 索引字段: user_id, memory_id, memory_type
- 距离度量: Cosine

---

## 6. Feed 系统

### 6.1 数据流

```
7种数据源 → SearchSourceManager并发抓取 (45s超时, 5条总数限制)
  → normalize_raw_item (数据标准化)
  → deduplicate_items (内容hash去重)
  → FeedScorer.score (4维评分)
  → generate_feed_card (卡片生成: title, one_sentence_value, information_gap)
  → mix_cards (30/40/30比例混排)
  → 持久化到 feed_cards + info_items 表
```

### 6.2 数据源详情

| 来源 | 默认状态 | 配置要点 |
|------|---------|---------|
| ManualSeed | 启用 | 手动种子数据, 也是最终fallback |
| GitHub | 启用 | topics: agent, rag, llm, langgraph, mcp; min_stars≥50; Python/TypeScript |
| Arxiv | 启用 | 类别: cs.AI, cs.CL, cs.LG; 查询词: agent, rag, multi-agent, tool use, llm |
| DuckDuckGo | 启用 | region: wt-wt; safesearch: moderate; time: w (一周) |
| RSS | 启用 | 默认空URL, 按配置加载 |
| Tavily | 启用 | 需TAVILY_API_KEY; depth: basic |
| SerpApi | 启用 | 需SERPAPI_API_KEY; engine: google; hl: zh-cn; gl: cn |

### 6.3 卡片混排 (mixer.py)

按relation_type分成三类, 按30%/40%/30%比例混排:
- **explicit** (30%): 直接匹配用户目标和兴趣
- **adjacent** (40%): 相邻领域扩展
- **far** (30%): 远距离信息差探索

去重策略: 低置信度卡片占比 ≤20%

---

## 7. RAG 系统

### 7.1 文档摄入流程

```
上传文件 (PDF/DOCX/TXT/MD/CSV/XLSX/HTML, ≤20MB)
  → parse_document (转Markdown)
  → chunk_markdown (语义分块, ~500 tokens/chunk)
  → embed_texts (384维, DashScope text-embedding-v4)
  → QdrantVectorStore.upsert_chunks (集合: agent_os_documents)
  → 写入 document_chunks 表 (chunk_index, content, qdrant_point_id)
```

### 7.2 检索接口

- `rag_service.search()`: 纯向量检索, 返回top_k + score (user_id过滤)
- `rag_service.ask()`: 检索 + 提取式回答 (无LLM, 拼接前3条chunk内容)
- `rag_service.search_evidence()`: 轻量证据检索 (供context_builder注入GSSC)

### 7.3 Qdrant 文档存储

- 集合: `agent_os_documents`
- 维度: 384, 距离: Cosine, 超时: 30s
- 过滤: user_id (必须), document_id (可选)
- 支持 Qdrant Cloud 新版 `query_points` API

---

## 8. 图片识别系统 ★ (最新完成)

文件: `src/web_app/services/qwen_multimodal_service.py`, `src/web_app/services/agent_service.py`

### 8.1 触发条件

在 `agent_service.run_agent_async()` 中最先检查: `_is_direct_image_question(user_input, attachments)`

判断标准 (优先级):
1. 附件中有 kind=image 的文件
2. 用户消息包含图片分析关键词 (中英文20+个: "分析图片", "analyze this image", "这是什么", "图片里"等)
3. 短消息 (≤30字) + 有图片 → 自动视为图片问题
4. 但如果同时有文档附件且用户提到了文档关键词 → 走正常Agent流程

### 8.2 执行路径

如果判定为直接图片问题 → **快速通道 (跳过整个Agent Runtime)**:
1. 调用 `qwen_multimodal_service.answer_image_question()` 获取自然语言回答
2. 直接持久化 assistant_message (标记 `direct_image_answer: true`)
3. 直接推送SSE事件 (answer_started → answer_delta → answer_completed)
4. 不启动LangGraph图

如果非直接图片问题 (有文档+图片混合) → 正常Agent流程, 图片分析结果注入 attachment_context

### 8.3 多模态配置

- 模型: `qwen_vision_model` (默认 qwen3.6-plus, 支持 qwen-vl 系列)
- 限制: 单图 ≤10MB, 每次最多5张, 总base64 ≤40MB
- 两种模式:
  - `analyze_images()`: 返回结构化内部上下文 (供RAG/attachment_context)
  - `answer_image_question()`: 返回用户可读的自然语言回答
- 输出清理: `_clean_direct_image_answer()` 移除以 `[Image Understanding]`, `Image:`, `Description:`, `OCR:` 等开头的内部标签行

---

## 9. SSE 事件系统

文件: `src/web_app/agent/runtime/events.py`

### 9.1 事件类型与 Channels

| 事件 | 可见性 | display_channel | 说明 |
|------|--------|-----------------|------|
| visible_thought_delta | user | thinking | Agent思考进度 (自然语言) |
| visible_progress_delta | user | thinking | 进度更新 (百分比/步骤) |
| answer_started | user | answer | 回答开始 (SSE stream打开) |
| answer_delta | user | answer | 流式回答片段 |
| answer_completed | user | answer | 回答完成 |
| tool_call_started/delta/completed/failed | user | tool | 工具调用生命周期 |
| approval_required/granted/rejected | user | status | 审批状态变更 |
| run_created/completed/failed/paused/resumed | user | status | 运行状态变更 |
| milestone_started/completed | user | thinking | 里程碑标记 |

### 9.2 SSE 格式

```
event: {event_type}
data: {json_string}

```

前端根据 `display_channel` 决定渲染到哪个UI区域 (thinking面板 / answer面板 / tool面板 / status栏)

---

## 10. LLM 配置体系

文件: `src/web_app/agent/llm/config.py`, `src/web_app/agent/llm/router.py`

### 10.1 模型分层 (按 purpose)

| Purpose | 默认模型 | Tier | 用途 |
|---------|---------|------|------|
| intent | qwen3.6-flash | fast | 意图识别 (低延迟) |
| safety | qwen3.6-flash | fast | 安全检查 |
| memory | qwen3.6-flash | fast | 记忆提取 |
| skill | qwen3.6-flash | fast | Skill匹配 |
| planner | qwen3.6-max-preview | balanced | 规划推理 |
| rag | qwen3.6-max-preview | balanced | RAG查询理解 |
| artifact | qwen3.6-plus | balanced | 成果物生成 |
| final | qwen3.6-plus | balanced | 最终回答 |
| research | qwen3.7-plus | strong | 深度研究 (最强) |
| agent_llm | qwen3.6-plus | balanced | Agent默认 |
| vision | qwen3.6-plus | balanced | 多模态图片理解 |
| embedding | text-embedding-v4 | fast | 文本嵌入 (384维) |

### 10.2 Provider 架构

- `aliyun` (默认): DashScope API → `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - 通过 `langchain-openai` 的 `ChatOpenAI` 兼容接口调用
- `openai_compatible`: 通用OpenAI兼容API
- `disabled`: 完全关闭LLM (调试用)
- Factory模式: `ChatOpenAI` 实例工厂, 统一 timeout(60s) + max_retries(2) + temperature(0.2)

### 10.3 LLM 调用日志 (usage.py)

记录每次LLM调用: run_id, purpose, provider, model, latency_ms, input_tokens, output_tokens
可通过 `AGENT_LLM_USAGE_LOG_ENABLED` 开关, `AGENT_LLM_LOG_PROMPT_PREVIEW` 开启prompt预览

---

## 11. 数据库架构 (PostgreSQL)

### 11.1 核心表

| 表名 | 说明 | 关键字段 |
|------|------|---------|
| users | 用户 | id, email, hashed_password, nickname, avatar_url, status |
| user_profiles | 用户画像 | user_id, segment, goals(JSON), interests(JSON), feed_ratio_config |
| memories | 记忆 | user_id, memory_type(working/episodic/semantic/perceptual), content, importance(0-1), qdrant_point_id, metadata |
| feed_cards | Feed卡片 | user_id, info_item_id, title, one_sentence_value, information_gap, final_score, exposure_bucket, relation_type |
| info_items | 信息条目 | title, content, source_url, content_hash, source_type, published_at |
| documents | 文档 | user_id, filename, file_path, file_type, file_size, status, metadata |
| document_chunks | 文档分块 | document_id, chunk_index, content, qdrant_point_id, token_count |
| agent_runs | Agent运行 | user_id, conversation_id, thread_id, status(running/completed/failed), final_answer, langgraphstatus |
| agent_conversations | 会话 | user_id, conversation_id(uuid), title, status, thread_id |
| agent_chat_messages | 聊天消息 | conversation_id, role(user/assistant/system), content, metadata |
| agent_steps | 步骤记录 | run_id, node_name, input, output, status |
| agent_events | SSE事件 | run_id, event_type, node_name, payload |
| llm_calls | LLM调用日志 | run_id, purpose, provider, model, latency_ms, input_tokens, output_tokens |
| research_runs | 研究运行 | id(uuid), feed_card_id, query, status, findings, markdown_report |
| skills | 技能 | user_id, name, trigger_text, tool_plan, safety_level, status(draft/approved/disabled) |

### 11.2 辅助表

| 表名 | 说明 |
|------|------|
| approvals | 审批记录 (pending/approved/rejected) |
| artifacts | 成果物 (markdown文件路径+元数据) |
| tool_calls | MCP工具调用记录 |
| mcp_servers | MCP服务器配置 |
| mcp_tools | MCP工具注册 |
| info_sources | Feed数据源配置 |
| feed_feedback | 用户对Feed卡片的反馈 (thumbs_up/down, dismissed) |
| eval_records | 质量评估记录 |

---

## 12. 关键设计模式

### 12.1 Repository 模式
所有数据库操作通过 `db/repositories/` 封装, BaseRepository提供通用CRUD, 每个模型对应一个子类

### 12.2 Service 单例模式
业务逻辑Service在模块级别实例化为单例: `xxx_service = XxxService()`

### 12.3 多层降级策略 (Fallback Chain)
- **意图识别**: LLM → 规则引擎 (100+关键词)
- **LangGraph加载**: 正常图 → fallback顺序节点执行
- **记忆检索**: Qdrant语义搜索 → PostgreSQL ILIKE → 最近重要记忆
- **研究执行**: 上游ODR Adapter → FallbackResearcher (模拟数据)
- **最终回答**: LLM流式生成 → 规则兜底文案
- **Feed**: 7个真实源 → ManualSeed种子数据

### 12.4 非阻塞设计
- RAG搜索失败不阻断Agent流水线
- 记忆提取失败不阻断Agent Run
- 图片分析失败返回友好错误信息 (不抛异常)
- Feed刷新单个源超时/失败不影响其他源

### 12.5 安全防护
- 权限守卫节点在所有处理之前执行
- L3/L4操作需要审批 (approvals表)
- `final_response` 防护内部JSON泄露给普通用户
- JWT认证 (python-jose) + bcrypt密码哈希

---

## 13. API 端点总览

| 前缀 | 用途 | 关键端点 |
|------|------|---------|
| /api/v1/agent | Agent运行 | `POST /runs/stream` (SSE), CRUD conversations, messages |
| /api/v1/research | 研究 | `POST /runs` (异步后台), `GET /runs/:id` |
| /api/v1/documents | 文档+RAG | `POST /upload`, `POST /ingest`, `POST /rag/search`, `POST /rag/ask` |
| /api/v1/memory | 记忆 | `POST /add`, `POST /search`, `POST /consolidate`, `POST /forget`, `GET /growth-profile` |
| /api/v1/artifacts | 成果物 | CRUD + download |
| /api/v1/skills | 技能 | CRUD + `POST /:id/approve`, `POST /:id/disable` |
| /api/v1/mcp | MCP | `GET /tools`, `POST /execute` |
| /api/v1/approvals | 审批 | `POST /:id/approve`, `POST /:id/reject` |
| /api/v1/auth | 认证 | `POST /login`, `POST /register` |
| /api/v1/profile | 画像 | `GET /`, `PUT /` |
| /api/v1/feed | Feed | `GET /home`, `POST /refresh`, `POST /cards/:id/research` |
| /api/health | 健康检查 | `GET /` |

---

## 14. 上游开源项目 (src/open_deep_research/) 架构

### 14.1 主工作流 (deep_researcher.py)

```
START → clarify_with_user ──(need_clarification?)──→ END (返回问题)
           │ (no)
           ▼
      write_research_brief (生成研究简报)
           │
           ▼
      research_supervisor (子图: supervisor ⇄ supervisor_tools)
           │                    │
           │                    ├─ think_tool (战略反思)
           │                    ├─ ConductResearch → researcher子图 (并行执行)
           │                    └─ ResearchComplete → 结束
           ▼
      final_report_generation (综合报告生成)
           │
           ▼
          END
```

### 14.2 Researcher 子图 (每个并行实例)

```
START → researcher ⇄ researcher_tools → compress_research → END
```

- **researcher**: 使用搜索工具(配置的search_api) + think_tool进行研究
- **researcher_tools**: 并行执行所有tool_call, 支持Tavily/OpenAI/Anthropic/MCP
- **compress_research**: 压缩研究成果, 去重+整理, 保留全部信息来源

### 14.3 配置要点 (configuration.py)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| search_api | tavily | 搜索API (tavily/openai/anthropic/none) |
| max_concurrent_research_units | 5 | 最大并行Researcher数 |
| max_researcher_iterations | 6 | Supervisor最大反思轮次 |
| max_react_tool_calls | 10 | 单个Researcher最大工具调用数 |
| allow_clarification | True | 是否启用用户澄清环节 |
| summarization_model | openai:gpt-4.1-mini | 网页摘要模型 |
| research_model | openai:gpt-4.1 | 研究执行模型 |
| compression_model | openai:gpt-4.1 | 研究压缩模型 |
| final_report_model | openai:gpt-4.1 | 最终报告模型 |

### 14.4 Token管理 (utils.py)

- `MODEL_TOKEN_LIMITS` 字典: 覆盖 OpenAI, Anthropic, Google, Cohere, Mistral, Ollama, Bedrock 等主要模型
- `is_token_limit_exceeded()`: 按提供商模式匹配识别token超限错误
- `remove_up_to_last_ai_message()`: 超限时截断到最后一个AI消息之前
- 最终报告生成: 3次渐进式截断重试 (首次4×token_limit字符, 之后每次减10%)

---

## 15. 提交历史 (最近10次)

```
2406e69 ← HEAD: 完成了信息差卡片基本情况，下一步操作电脑
4e7d70e: 完成了基本搜索展示，下一步从多方搜
09a8c5f: 下一步修远域
fc60ce1: 完成了识图和文件
6d5cacd: 识图完成，下一步文档bug
815eb33: 完成了图片识别，下一步sse输出
cbb5fa2: 完成qdrant，下一步rag
1e2611a: 完成了基本记忆完善，下一步是markdown前端渲染
1d34c14: 完成记忆，下一步修复LLM识别回答问题
d99bd42: 下一步学codex
a3f3917: 下一步修bug
40007dd: 完成基本对话。下一步优化流式输出
c84bdde: 完成了对话的基本，但是输出还是有问题，下一步解决输出和思考过程的问题
49c5274: 完成会话管理，下一步要求正常对话
c3b38e5: 完成基本搭建
eff3b0b: 完成信息接入
7b4b043: 完成基本agent流转
8769c14: 完成基本第一轮架构
1dfce8e: 完成基本架构
904c0c2: Initial commit
```

---

## 16. 当前状态总结

### 已完成
- 基础Agent框架 (LangGraph 14节点多智能体图)
- 会话管理 (CRUD, 多轮对话, agent_conversations/agent_chat_messages)
- Feed系统 (7源聚合, 评分, 混排, 信息差卡片)
- 记忆系统 (三层架构 + Qdrant语义检索 + 确定性提取器)
- RAG系统 (文档上传/解析/分块/摄入/检索问答)
- 图片识别 (Qwen多模态, 直接问答快速通道, 混合附件处理)
- SSE流式输出 (4 channel: thinking/answer/tool/status)
- Skill系统 (匹配/创建/审批/自动草稿)
- MCP工具系统 (注册/执行/审批/审计)
- LLM配置体系 (3 tier × 10 purpose, aliyun DashScope为主)
- 多层降级策略 (LLM→规则, Qdrant→PostgreSQL→recent, 上游→fallback)

### 已知待优化
- SSE流式输出体验优化
- Markdown前端渲染
- LLM回答质量 (有时输出JSON而非自然语言, 已有防护但需改进)
- Research Adapter主要是mock, 真实调用上游需配置
- Codex风格的界面布局
- 远域 (remote domain) 功能
- 文档bug修复

### 用户约束 (从项目中记忆提取)
- 不要修改 `src/open_deep_research/` 目录 (上游原始项目)
- 不要引入 Neo4j (已有配置但 `ENABLE_NEO4J=false`)
- 不重写Agent Runtime核心, 不破坏现有API
- 偏好中文表达, 简洁, 产品化风格
- 不要把内部JSON暴露给普通用户

---

## 17. 开发命令

```bash
# 环境准备
cp .env.example .env
# 编辑 .env 配置数据库连接、LLM API key等

# 启动服务
uvx langgraph dev                         # LangGraph Studio (上游)
uvicorn src.web_app.main:app --reload     # FastAPI Agent OS (主服务)

# 代码质量
ruff check                                # linting (E, F, I, D, UP规则)
mypy                                      # type checking

# 测试
python tests/run_evaluate.py              # LangSmith评估
pytest                                    # 单元测试
```
