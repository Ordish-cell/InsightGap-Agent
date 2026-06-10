# Open Deep Research — Agent OS 项目全量上下文

> 本文档供 AI 助手快速建立对该项目的完整理解。最后更新: 2026-06-10。

---

## 1. 项目概览

本项目基于开源项目 [Open Deep Research](https://github.com/langchain-ai/open_deep_research) (v0.0.16) 进行二次开发，构建了一个**信息差 Agent OS** 平台。核心闭环：

```
Feed (信息流接入) → Agent 研究 → Artifact (成果物) → Memory/Skill (记忆沉淀)
```

| 维度 | 详情 |
|------|------|
| **技术栈** | Python 3.10+, FastAPI, LangGraph, LangChain, PostgreSQL, Qdrant, Redis |
| **LLM** | 阿里云 DashScope (百炼) Qwen 系列为主，兼容 OpenAI/Anthropic/Google/DeepSeek |
| **前端** | Vite + React 19 + TypeScript (独立项目，端口 localhost:5173) |
| **作者** | Ordish |
| **当前分支** | `feature/bendicaozuo` |
| **上游来源** | https://github.com/langchain-ai/open_deep_research |
| **所有分支** | `main`, `feature/bendicaozuo`, `feature/backend-step-1`, `feature/backend-step-2`, `feature-xinxicha` |

---

## 2. 目录结构（关键文件）

```
open_deep_research/
├── src/
│   ├── open_deep_research/          # 【上游开源项目 - 保留不改】
│   │   ├── deep_researcher.py       # ★ LangGraph supervisor-worker 研究主流程 (719行)
│   │   │   │                        #   节点: clarify_with_user → write_research_brief
│   │   │   │                        #        → supervisor子图 → final_report_generation
│   │   │   │                        #   Supervisor委托ConductResearch, Researcher子图并行执行
│   │   │   ├── state.py             # AgentState, SupervisorState, ResearcherState + 结构化输出
│   │   │   ├── configuration.py     # Configuration (Pydantic) — 所有可配置参数
│   │   │   ├── prompts.py           # 全部LLM提示词模板 (9个prompt)
│   │   │   └── utils.py             # 工具函数: Tavily搜索, MCP, think_tool, token管理 (926行)
│   │   │
│   │   ├── web_app/                 # 【二开核心 - Agent OS 平台】
│   │   │   ├── main.py              # FastAPI 入口 (title="Open Deep Research Agent OS API")
│   │   │   │
│   │   │   ├── core/
│   │   │   │   ├── config.py        # Settings (pydantic-settings, 170+配置项, .env加载)
│   │   │   │   ├── constants.py     # 风险等级常量 (L0-L4)
│   │   │   │   ├── errors.py        # 自定义异常类
│   │   │   │   ├── logging.py       # 日志配置
│   │   │   │   └── security.py      # 安全工具函数
│   │   │   │
│   │   │   ├── api/v1/              # 12 个 API 路由模块
│   │   │   │   ├── router.py        # APIRouter 汇总
│   │   │   │   ├── agent.py         # Agent运行 (SSE流式/非流式, 含审批恢复端点)
│   │   │   │   ├── research.py      # 深度研究 (异步后台任务)
│   │   │   │   ├── memory.py        # 记忆增删查改+固化
│   │   │   │   ├── documents.py     # 文档上传+RAG搜索
│   │   │   │   ├── artifacts.py     # 成果物管理
│   │   │   │   ├── skills.py        # Skill管理
│   │   │   │   ├── mcp.py           # MCP工具
│   │   │   │   ├── approvals.py     # 审批管理
│   │   │   │   ├── auth.py          # 认证 (login/register)
│   │   │   │   ├── profile.py       # 用户画像
│   │   │   │   ├── feed.py          # Feed流接口
│   │   │   │   └── health.py        # 健康检查 /api/health
│   │   │   │
│   │   │   ├── services/            # ★ 业务逻辑层 (单例模式, 20+ services)
│   │   │   │   ├── agent_service.py     # 【核心】Agent运行主流程 (含图片快速通道、审批暂停/恢复)
│   │   │   │   ├── research_service.py  # 深度研究编排 (adapter/fallback)
│   │   │   │   ├── memory_service.py    # 记忆提取+存储+检索 (三级回退, Jaccard去重)
│   │   │   │   ├── rag_service.py       # RAG检索问答 (含轻量证据检索)
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
│   │   │   │   ├── source_service.py    # Feed数据源管理
│   │   │   │   ├── artifact_service.py  # 成果物生成保存
│   │   │   │   └── conversation_lock.py # 会话并发锁 (防止同一会话并发执行)
│   │   │   │
│   │   │   ├── agent/                # Agent运行时 (LangGraph多智能体)
│   │   │   │   ├── state.py          # AgentState TypedDict (20+字段)
│   │   │   │   ├── schemas.py        # AgentRunRequest Pydantic模型
│   │   │   │   ├── graph.py          # 简单函数管道 (备用)
│   │   │   │   ├── runtime/          # 【核心运行时】
│   │   │   │   │   ├── __init__.py   # 导出AgentRuntime
│   │   │   │   │   ├── graph.py      # ★ LangGraph图定义 (含 true interrupt: approval→END)
│   │   │   │   │   ├── nodes.py      # ★ RuntimeNodes类 (15个节点方法, 含LLM意图/知识库关键词匹配)
│   │   │   │   │   ├── router.py     # 条件路由 dispatch_next_route_node (支持 END sentinel)
│   │   │   │   │   ├── planner.py    # 确定性关键字路由 plan_route() + 强制路由
│   │   │   │   │   ├── state.py      # AgentRuntimeState TypedDict + AgentIntent枚举 (17种)
│   │   │   │   │   ├── emitter.py    # ★ RuntimeEventEmitter — 统一SSE事件发射器 (273行)
│   │   │   │   │   ├── events.py     # SSE事件类型 + display_channel定义
│   │   │   │   │   ├── visible_thoughts.py # 用户可见思考进度
│   │   │   │   │   ├── visibility.py # 可见性检查
│   │   │   │   │   ├── intent_llm.py # ★ 基于LLM的意图识别 (LLM+规则双引擎, 124行)
│   │   │   │   │   ├── intent_schema.py  # 意图结构定义 (HomeIntentResult + LLMToolSelectionResult, 143行)
│   │   │   │   │   ├── fallback.py   # 兜底逻辑 (无LangGraph时)
│   │   │   │   │   ├── checkpoint.py # 事件记录
│   │   │   │   │   ├── checkpointers.py # LangGraph checkpoint
│   │   │   │   │   ├── langgraph_status.py # 执行状态追踪 (max_steps=12)
│   │   │   │   │   └── progress.py   # 进度管理
│   │   │   │   ├── llm/              # LLM配置层
│   │   │   │   │   ├── config.py     # LLMSettings
│   │   │   │   │   ├── router.py     # 按purpose路由到具体模型 (intent→fast, research→strong等)
│   │   │   │   │   ├── factory.py    # ChatOpenAI兼容实例工厂
│   │   │   │   │   ├── embedding.py  # Embedding模型配置 (DashScope/OpenAI)
│   │   │   │   │   ├── errors.py     # LLM相关异常 (LLMUnavailableError, LLMParseError, LLMInvocationError)
│   │   │   │   │   └── usage.py      # LLM调用日志记录
│   │   │   │   ├── adapters/         # 外部适配器
│   │   │   │   │   ├── mcp_adapter.py
│   │   │   │   │   └── open_deep_research_adapter.py  # 上游ODR适配 (334行+)
│   │   │   │   ├── nodes/            # 备用Agent节点 (骨架实现)
│   │   │   │   └── planners/         # 旧版规划器 (react/plan_and_solve)
│   │   │   │
│   │   │   ├── models/               # SQLAlchemy ORM 模型 (25+ 表)
│   │   │   │   ├── orm.py            # ★ 主ORM定义
│   │   │   │   ├── agent_run.py, approval.py, artifact.py, document.py
│   │   │   │   ├── entities.py, feed_card.py, feed_feedback.py, info_item.py
│   │   │   │   ├── memory.py, profile.py, research_run.py, skill.py
│   │   │   │   ├── source.py, tool.py, user.py, eval_record.py
│   │   │   │
│   │   │   ├── db/
│   │   │   │   ├── base.py           # SQLAlchemy Base声明
│   │   │   │   ├── session.py        # DB session工厂 (SessionLocal)
│   │   │   │   ├── init_db.py        # 建表脚本
│   │   │   │   └── repositories/     # 13+ Repository (数据访问层)
│   │   │   │       ├── base_repository.py
│   │   │   │       ├── agent_repository.py, approval_repository.py
│   │   │   │       ├── artifact_repository.py, document_repository.py
│   │   │   │       ├── feed_repository.py, info_repository.py
│   │   │   │       ├── mcp_repository.py, memory_repository.py
│   │   │   │       ├── profile_repository.py, research_repository.py
│   │   │   │       ├── skill_repository.py, user_repository.py
│   │   │   │
│   │   │   ├── memory/               # 记忆子系统 (三层架构 + LLM提取)
│   │   │   │   ├── base.py           # BaseMemoryStore抽象类
│   │   │   │   ├── working.py        # 工作记忆 (Redis TTL=3600s)
│   │   │   │   ├── episodic.py       # 情景记忆 (具体事件)
│   │   │   │   ├── semantic.py       # 语义记忆 (长期偏好/知识)
│   │   │   │   ├── perceptual.py     # 感知记忆 (原始感知数据)
│   │   │   │   ├── extractor.py      # ★ 双重记忆提取器 — MemoryExtractor (正则) + LlmMemoryExtractor (LLM, 使用qwen3.6-max-preview)
│   │   │   │   ├── consolidation.py  # 记忆固化 (importance阈值判断)
│   │   │   │   └── qdrant_memory_store.py # Qdrant向量存储 (memory_vectors集合, 384维)
│   │   │   │
│   │   │   ├── feed/                 # Feed系统 (信息差)
│   │   │   │   ├── sources/          # 7+1种数据源
│   │   │   │   │   ├── manager.py    # SearchSourceManager (统一调度, 45s超时)
│   │   │   │   │   ├── arxiv.py, github.py, rss.py
│   │   │   │   │   ├── duckduckgo.py, tavily.py, serpapi.py
│   │   │   │   │   ├── bucket_seed.py  # 桶种子数据
│   │   │   │   │   └── manual_seed.py  # 手动种子数据 (final fallback)
│   │   │   │   ├── scorer.py         # FeedScorer 评分引擎 (4维评分)
│   │   │   │   ├── normalizer.py     # 数据标准化
│   │   │   │   ├── dedup.py          # 去重逻辑
│   │   │   │   ├── mixer.py          # 卡片混排 (30/40/30比例)
│   │   │   │   └── card_generator.py # 卡片生成 (title, one_sentence_value, information_gap)
│   │   │   │
│   │   │   ├── rag/                  # RAG系统
│   │   │   │   ├── vector_store.py   # QdrantVectorStore (Qdrant Cloud API)
│   │   │   │   ├── embeddings.py     # 嵌入生成 (DashScope/SentenceTransformer, 384维)
│   │   │   │   ├── chunker.py        # Markdown语义分块 (~500 tokens/chunk)
│   │   │   │   ├── chunking.py       # 简单文本分块
│   │   │   │   ├── document_parser.py # PDF/DOCX/TXT/MD/CSV/XLSX/HTML解析 (markitdown + fallback)
│   │   │   │   ├── document_loader.py # 简单文本加载器
│   │   │   │   ├── retriever.py      # 检索器存根
│   │   │   │   └── qdrant_client.py  # Qdrant连接配置/健康检查
│   │   │   │
│   │   │   ├── mcp/                  # MCP工具系统
│   │   │   │   ├── registry.py       # 工具注册中心
│   │   │   │   ├── tool_executor.py  # 工具执行 (含审批检查)
│   │   │   │   ├── tool_router.py    # ★ 工具路由 (LLM+关键词双引擎匹配, 292行+)
│   │   │   │   ├── local_provider.py # ★ 本地工具Provider (本地文件+邮件+记忆+成果物, 119行+)
│   │   │   │   ├── email_provider.py # ★ 邮件工具Provider (Mock + SMTP, 中英文别名)
│   │   │   │   ├── local_file_tools.py # ★ 本地文件操作工具 (workspace沙箱, 敏感文件阻断, 198行+)
│   │   │   │   ├── permissions.py    # 权限控制
│   │   │   │   ├── schemas.py        # 数据结构 (含工具注册schema)
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
│   │   │   │   ├── report_builder.py, evidence_builder.py, schemas.py
│   │   │   │
│   │   │   ├── artifacts/            # 成果物
│   │   │   │   ├── generators.py     # ArtifactGenerator
│   │   │   │   └── storage.py        # 文件存储 (本地磁盘)
│   │   │   │
│   │   │   ├── schemas/              # 通用schema
│   │   │   │   └── common.py         # 通用响应模型
│   │   │   │
│   │   │   └── tests/                # ★ 单元测试 (新增)
│   │   │       ├── test_memory_system.py  # 记忆系统综合测试 (560行)
│   │   │       └── db_test_utils.py       # 测试数据库工具
│   │   │
│   │   ├── security/
│   │   │   └── auth.py               # LangGraph部署认证 (Supabase JWT)
│   │   │
│   │   └── legacy/                   # 旧版实现 (不再活跃开发)
│   │       ├── graph.py              # Plan-and-execute + human-in-the-loop
│   │       └── multi_agent.py        # Supervisor-researcher多Agent
│   │
│   ├── tests/                        # 评估框架 + 集成测试
│   │   ├── run_evaluate.py           # LangSmith评估主脚本
│   │   ├── evaluators.py             # 6个评估器
│   │   ├── prompts.py                # 评估提示词
│   │   ├── pairwise_evaluation.py    # 配对比较评估
│   │   ├── supervisor_parallel_evaluation.py  # 多线程并行评估
│   │   ├── extract_langsmith_data.py # LangSmith数据导出
│   │   ├── test_approval_workflow.py # ★ 审批工作流集成测试 (1148行)
│   │   ├── test_llm_tool_selection.py    # ★ LLM工具选择测试 (302行)
│   │   ├── test_tool_input_validation.py # ★ 工具输入验证测试 (99行)
│   │   ├── test_tool_name_normalization.py # ★ 工具名规范化测试 (73行)
│   │   └── expt_results/             # 实验结果 (JSONL)
│   │
│   ├── examples/                     # 研究示例 (arxiv, pubmed, inference-market)
│   │
│   ├── scripts/
│   │   ├── ensure_qdrant_indexes.py   # Qdrant索引初始化脚本
│   │   └── backfill_memory_vectors.py # 记忆向量回填脚本
│   │
│   ├── alembic/                      # 数据库迁移
│   │   └── versions/
│   │       ├── 20260608_0007_feed_batch.py
│   │       └── 20260608_0008_feed_refresh_attempt.py
│   │
│   ├── frontend/                     # React前端 (独立Vite项目)
│   │   ├── src/
│   │   │   ├── main.tsx              # React入口
│   │   │   ├── App.tsx               # 根组件+路由
│   │   │   ├── api/                  # API层 (13个模块)
│   │   │   │   ├── client.ts         # HTTP客户端
│   │   │   │   ├── agent.ts          # Agent SSE流式API
│   │   │   │   ├── research.ts, feed.ts, documents.ts, memory.ts
│   │   │   │   ├── artifacts.ts, skills.ts, mcp.ts, approvals.ts
│   │   │   │   ├── auth.ts, health.ts, types.ts, normalizers.ts
│   │   │   ├── pages/                # 13个页面
│   │   │   │   ├── HomePage.tsx, AgentRunPage.tsx, FeedPage.tsx
│   │   │   │   ├── FeedCardDetailPage.tsx, ResearchRunsPage.tsx
│   │   │   │   ├── ResearchRunDetailPage.tsx, MemoryPage.tsx
│   │   │   │   ├── ArtifactsPage.tsx, SkillsPage.tsx, McpToolCallsPage.tsx
│   │   │   │   ├── ApprovalsPage.tsx, LoginPage.tsx, ProfilePage.tsx
│   │   │   │   └── SettingsPage.tsx
│   │   │   ├── components/
│   │   │   │   ├── agent/  (AgentChatPanel, AgentThoughtStream, ApprovalCard)
│   │   │   │   ├── common/ (MarkdownRenderer, JsonBlock, StatusPill, 等)
│   │   │   │   └── layout/ (AppShell, Sidebar, Topbar)
│   │   │   └── styles/
│   │   │       └── global.css        # 全局样式 (219行+)
│   │   └── package.json              # React 19, react-markdown, react-router-dom
│   │
│   ├── langgraph.json                # LangGraph部署配置 (入口: deep_researcher)
│   ├── pyproject.toml                # 项目依赖 (60+包) + ruff/mypy配置
│   ├── .env.example                  # 环境变量模板
│   ├── .mcp.json                     # MCP服务器配置
│   ├── CLAUDE.md                     # Claude Code项目指令
│   └── PROJECT_CONTEXT.md            # 本文档
```

---

## 3. 核心架构：Agent 运行时

### 3.1 请求生命周期

```
HTTP POST /api/v1/agent/runs/stream (SSE流式)
  └── agent_service.run_agent_async()
        ├── 1. 创建/获取会话 (agent_conversations表)
        ├── 2. 创建AgentRun记录 (status: running)
        ├── 3. 处理附件 (图片→多模态分析, 文档→RAG摄入)
        ├── 4. 判断是否直接图片问答 ★ (fast path, 跳过LangGraph)
        ├── 5. → AgentRuntime.run() 启动LangGraph多智能体图
        │     └── [审批触发时] → 图中断(END sentinel) → 等待用户审批 → resume_from_approval()
        └── 6. 持久化结果 + 推送SSE events (agent_events表)

审批恢复流程:
HTTP POST /api/v1/agent/runs/{id}/resume
  └── agent_service.resume_agent_run()
        └── AgentRuntime.resume_from_approval(state) → 清理挂起状态 → 重新执行图
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
      ├── tool_agent       (MCP工具推断+调用, L3/L4→触发审批→true interrupt→END)
      ├── memory_agent     (记忆提取+保存, 三层记忆)
      └── skill_agent      (Skill检测+自动创建草稿)
  → evaluator              (质量评估, 完整性检查)
  → final_response         (LLM流式生成最终回答, 内部JSON防护)
  → END
```

**True Interrupt 机制 (新增)**:
- `tool_agent` 执行 L3/L4 操作时, 设置 `status=waiting_approval`
- `dispatch_next_route_node` 检测到此状态, 直接返回 `__end__` 映射到 `END`
- 图干净终止, 不经过 evaluator / final_response
- `agent_service` 检测暂停状态, 发射 `approval_required` / `run_paused` 事件
- 用户审批后, `resume_from_approval()` 清理挂起状态, 重新调用 `run(state)`

多路由支持: route_plan.route 是列表, agent按序执行, 每个agent完成后router决定下一步(下一个agent/评估/最终回答/中断)

### 3.3 RuntimeEventEmitter (新增核心组件)

文件: `src/web_app/agent/runtime/emitter.py` (273行)

统一事件发射器,所有运行时节点使用同一个 emitter 实例,自动:
- 递增 event_seq
- 附加 run_id / conversation_id / message_id / created_at
- 推送到 SSE 队列 (queue_stream_event)
- 持久化到 agent_events 表 (record_event)

**便捷方法**:

| 方法 | display_channel | 事件类型 |
|------|-----------------|---------|
| `thought(text)` | thinking | visible_thought_delta |
| `answer_delta(text)` | answer | answer_delta |
| `answer_started()` | answer | answer_started |
| `answer_completed(answer)` | answer | answer_completed |
| `tool(event_type, payload)` | tool | tool_call_started/completed/failed |
| `status(event_type, payload)` | status | 通用状态事件 |
| `run_created(...)` | status | run_created |
| `run_completed(answer)` | status | run_completed |
| `run_failed(error)` | status | run_failed |
| `run_paused(approval_id)` | status | run_paused |
| `run_resumed()` | status | run_resumed |
| `approval_required(...)` | status | approval_required (含前端就绪payload) |
| `approval_granted(id)` | status | approval_granted |
| `approval_rejected(id)` | status | approval_rejected |

### 3.4 14个运行时节点

| 节点 | 核心逻辑 |
|------|---------|
| **permission_guard** | 关键词检测风险等级(L0-L4), L3+触发审批, blocked→final_response |
| **home_intent_react** | ★ LLM意图分类 (`intent_llm.infer_home_intent_with_llm`) + 规则引擎兜底, 输出AgentIntent枚举 (17种) |
| **planner** | `plan_route()`: 基于100+中英文关键词+强制路由+LLM意图+FeedCard+知识库关键词匹配, 输出RoutePlan |
| **context_builder** | GSSC四阶段: 收集→选择→结构化→压缩, 14种ContextPacket来源 |
| **skill_matcher** | 检索已批准Skill, 向量+关键词匹配, score≥0.75自动触发 |
| **research_agent** | 调用ResearchService, 搜索+分析+证据构建+Report+Skill草稿 |
| **rag_agent** | RAG检索用户知识库, 提取证据 |
| **artifact_agent** | 将research/rag结果保存为Markdown成果物 |
| **tool_agent** | ★ MCP工具名推断(ToolRouter: LLM+关键词双引擎)→工具执行(L3/L4→true interrupt→END)→结果注入上下文 |
| **memory_agent** | MemoryExtractor + LlmMemoryExtractor提取→去重(Jaccard)→保存(working/episodic/semantic)→同步Qdrant |
| **skill_agent** | 评估可复用性, 自动创建Skill草稿(status=draft) |
| **skill_draft_detector** | 检测是否需要创建Skill草稿 |
| **evaluator** | 评估执行结果完整性, 设置run最终状态 |
| **final_response** | LLM流式生成用户可读回答, 防护内部JSON泄露, 推送SSE answer_delta |

### 3.5 路由规划详细逻辑 (planner.py: plan_route())

**17种意图类型** (增补: `tool.local_file`, `tool.shell_readonly`, `tool.shell_write`, `tool.dangerous`):
`chat`, `research`, `feed_research`, `rag`, `artifact`, `tool`, `memory`, `skill`, `mixed`, `tool.email`, `tool.local_file`, `tool.browser`, `tool.comment`, `tool.form_submit`, `tool.shell_readonly`, `tool.shell_write`, `tool.dangerous`

**优先级体系**:
1. forced_route (payload中显式指定) → 最高优先
2. LLM意图识别结果 (home_intent_react)
3. FeedCard附带 → `feed_research`
4. 显式记忆写入关键词 ("记住"/"以后"/"别忘了") → `memory`, 覆盖 research/rag/artifact
5. 对话回忆检测 ("之前"/"上次"/"说过") → `chat` (不回退到研究)
6. 知识库关键词匹配 (知识库/文档/资料) → `rag`
7. 规则关键词匹配 (100+中英文关键词) → 按意图优先级排列
8. 默认 → `chat`

**风险等级**:
| 等级 | 说明 | 示例 | 行为 |
|------|------|------|------|
| L0 | 只读闲聊 | 对话, 解释 | 直接放行 |
| L1 | 搜索 | 搜索, RAG | 直接放行 |
| L2 | 生成 | 研究, 生成Artifact | 直接放行 |
| L3 | 外部写入 | 发送邮件, 评论, 文件操作 | 需要审批 → true interrupt |
| L4 | 破坏性 | 删除, 支付, 转账 | 需要审批 → true interrupt |

---

## 4. 上下文系统 (ContextBuilder — GSSC)

文件: `src/web_app/context/builder.py`

### 4.1 GSSC 四阶段

1. **Gather**: 收集14种来源的ContextPacket, 按路由设定relevance权重
2. **Select**: 按权重排序, 在token预算内选择
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

### 5.2 记忆提取器 — 双重引擎

文件: `src/web_app/memory/extractor.py`

**MemoryExtractor** (确定性, 纯正则规则):
- working: 当前页面, 选中的FeedCard ID
- episodic: Skill匹配/创建, 研究行为, 用户反馈 (正面/负面)
- semantic: 项目目标, 技术栈, 边界约束, 产品偏好, 信息兴趣

**LlmMemoryExtractor** (LLM驱动, 使用 qwen3.6-max-preview):
- 使用 Pydantic 结构化输出 (`MemoryExtractionResult`, `LongTermMemoryItem`)
- LLM 失败时自动回退到 MemoryExtractor 正则规则
- 中文提示词, 提取维度: project_goals, tech_stack, boundaries, preferences, feed_interests, feedback, skill_*, research_action, working_context

**去重规则**:
- 语义记忆写前检查: Jaccard相似度 ≥0.55 (字符+ngram双重) 视为重复, 合并+更新importance
- 对话闲谈 (问候/感谢/道别) 只保留 importance≥0.80 的语义记忆, 降权至 ≤0.50

### 5.3 记忆固化 (consolidation.py)

- working → episodic: importance ≥ 0.7
- episodic → semantic: importance ≥ 0.8

### 5.4 记忆检索 (MemoryService.search_memory)

三级回退策略:
1. **Qdrant 语义搜索** — score≥0.25, top 8, 按 `0.75 × Qdrant score + 0.25 × importance` 排序
2. **PostgreSQL ILIKE** — 关键词模糊匹配 (fallback)
3. **最近重要记忆** — importance≥0.7的semantic记忆, top 5 (final fallback)

### 5.5 记忆遗忘策略

- `forget_memory()`: 按ID删除 (立即)
- `forget_by_importance()`: 删除低于阈值的非受保护记忆
- `forget_by_time()`: 删除超过 retention_days 的非受保护记忆
- `forget_by_capacity()`: 超容量时按分数删除最不重要的记忆

### 5.6 Qdrant 记忆存储

- 集合: `memory_vectors` (可配置), 维度: 384, 距离: Cosine
- 内容截断: 最多嵌入前4000字符
- 索引字段: user_id, memory_id, memory_type

---

## 6. Feed 系统

### 6.1 数据流

```
7+1种数据源 → SearchSourceManager并发抓取 (45s超时, 5条总数限制)
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
| ManualSeed | 启用 | 手动种子数据, 最终fallback |
| BucketSeed | 启用 | 桶种子数据 (预置高质量信息) |
| GitHub | 启用 | topics: agent, rag, llm, langgraph, mcp; min_stars≥50 |
| Arxiv | 启用 | 类别: cs.AI, cs.CL, cs.LG |
| DuckDuckGo | 启用 | region: wt-wt; safesearch: moderate; time: w |
| RSS | 启用 | 默认空URL, 按配置加载 |
| Tavily | 启用 | 需TAVILY_API_KEY; depth: basic |
| SerpApi | 启用 | 需SERPAPI_API_KEY; hl: zh-cn; gl: cn |

### 6.3 卡片混排 (mixer.py)

按relation_type分成三类, 按30%/40%/30%比例混排:
- **explicit** (30%): 直接匹配用户目标和兴趣
- **adjacent** (40%): 相邻领域扩展
- **far** (30%): 远距离信息差探索

---

## 7. RAG 系统

### 7.1 文档摄入流程

```
上传文件 (PDF/DOCX/TXT/MD/CSV/XLSX/HTML, ≤20MB)
  → parse_document (markitdown → fallback专用解析器)
  → chunk_markdown (语义分块, ~500 tokens/chunk, 跟踪heading_path)
  → embed_texts (384维, DashScope text-embedding-v4)
  → QdrantVectorStore.upsert_chunks (集合: agent_os_documents)
  → 写入 document_chunks 表 (chunk_index, content, qdrant_point_id)
```

### 7.2 检索接口

- `rag_service.search()`: 纯向量检索, user_id过滤, 可选document_ids范围
- `rag_service.ask()`: 检索 + 提取式回答 (含ContextBuilder增强, 无证据时提取式fallback)
- `rag_service.ask_document()`: 文档特定Q&A (overview chunks + vector search混合)
- `rag_service.search_evidence()`: 轻量证据检索 (供context_builder注入GSSC)

### 7.3 Qdrant 文档存储

- 集合: `agent_os_documents`, 维度: 384, 距离: Cosine, 超时: 30s
- 过滤: user_id (必须), document_id (可选)
- 支持 Qdrant Cloud 新版 `query_points` API

---

## 8. MCP 工具系统 (含新增本地文件 + 邮件)

### 8.1 工具路由 (ToolRouter — LLM+关键词双引擎)

文件: `src/web_app/mcp/tool_router.py` (292行+)

1. **LLM 引擎** (`LLMToolSelectionResult`): 结构化输出, 同时选择多个工具
2. **关键词回退**: `_simple_tool_name_match()` — 中英文关键词覆盖所有已注册工具
3. **工具名规范化**: 40+ 别名映射 (send_email→email.send, 发邮件→email.send, etc.)
4. **参数推断**: `_infer_arguments()` — LLM提取 + 正则fallback
5. **缺失字段检测**: 工具参数不完整时返回 question 提示用户补充

### 8.2 本地文件操作 (新增)

文件: `src/web_app/mcp/local_file_tools.py` (198行+)

**工具列表**:

| 工具 | 风险 | 说明 |
|------|------|------|
| `local_file.read` | L1 | 读取文件内容, 限制 max_size_bytes |
| `local_file.write` | L3 | 写入文件 (需审批) |
| `local_file.append` | L3 | 追加到文件 (需审批) |
| `local_file.list` | L1 | 列出目录内容 |
| `local_file.delete` | L3/L4 | 删除文件 (高风险, 需审批) |

**安全措施**:
- 所有操作限制在 `LOCAL_TOOLS_WORKSPACE_DIR` 沙箱内
- 敏感文件模式阻断 (.env, .git, id_rsa, *.pem, *.key, *secret*, *token*, *password*, *credential*, *.pfx 等)
- 路径穿越检测 (`..` 和绝对路径)
- 读写大小限制

### 8.3 邮件工具 (新增)

文件: `src/web_app/mcp/email_provider.py` (108行+)

- **MockEmailProvider**: 仅写审计日志, 不实际发送 (默认)
- **SMTPProvider**: 真实SMTP发送 (需配置 SMTP_HOST/PORT/USER/PASSWORD)
- **中英文别名**: `email.send` ↔ `发送邮件`/`发邮件`/`send_email`/`sendEmail`/`寄邮件` 等15+种
- **安全**: SMTP密码绝不记录到日志

---

## 9. 图片识别系统

文件: `src/web_app/services/qwen_multimodal_service.py`

### 9.1 触发条件

在 `agent_service.run_agent_async()` 中最先检查。判断标准 (优先级):
1. 附件中有 kind=image 的文件
2. 用户消息包含图片分析关键词 (中英文20+个)
3. 短消息 (≤30字) + 有图片 → 自动视为图片问题
4. 但同时有文档附件+用户提到文档关键词 → 走正常Agent流程

### 9.2 执行路径

如果是直接图片问题 → **快速通道 (跳过整个Agent Runtime)**:
- 直接调用 `qwen_multimodal_service.answer_image_question()`
- 不启动LangGraph图, 直接推送SSE

如果非直接图片问题 → 正常Agent流程, 图片分析结果注入 attachment_context

### 9.3 多模态配置

- 模型: `qwen_vision_model` (默认 qwen3.6-plus)
- 限制: 单图 ≤10MB, 每次最多5张, 总base64 ≤40MB
- 输出清理: `_clean_direct_image_answer()` 移除内部标签行

---

## 10. SSE 事件系统

文件: `src/web_app/agent/runtime/events.py` + `emitter.py`

### 10.1 Channels

| display_channel | 事件 | UI区域 |
|-----------------|------|--------|
| **thinking** | visible_thought_delta, visible_progress_delta, milestone_* | 思考面板 |
| **answer** | answer_started, answer_delta, answer_completed | 回答面板 |
| **tool** | tool_call_started/delta/completed/failed | 工具面板 |
| **status** | approval_*, run_* | 状态栏 |

### 10.2 SSE 格式

```
event: {event_type}
data: {json_string}

```

### 10.3 统一发射器模式

所有节点使用 `RuntimeEventEmitter` 单实例, 不再在每个节点中手写 `queue_stream_event` + `record_event`, 确保:
- 事件序列号全局递增
- 所有事件附带 run_id/conversation_id/message_id/created_at
- DB持久化+SSE推送一步完成

---

## 11. LLM 配置体系

### 11.1 模型分层 (按 purpose)

| Purpose | 默认模型 | 用途 |
|---------|---------|------|
| intent | qwen3.6-max-preview | 意图识别 (低延迟) |
| safety | qwen3.6-max-preview | 安全检查 |
| memory | qwen3.6-max-preview | 记忆提取 |
| skill | qwen3.6-max-preview | Skill匹配 |
| planner | qwen3.6-max-preview | 规划推理 |
| rag | qwen3.6-max-preview | RAG查询理解 |
| artifact | qwen3.6-plus | 成果物生成 |
| final | qwen3.6-plus | 最终回答 |
| research | qwen3.7-plus | 深度研究 (最强) |
| agent_llm | qwen3.6-plus | Agent默认 |
| vision | qwen3.6-plus | 多模态图片理解 |
| embedding | text-embedding-v4 | 文本嵌入 (384维) |

### 11.2 Provider 架构

- `aliyun` (默认): DashScope API → `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - 通过 `langchain-openai` 的 `ChatOpenAI` 兼容接口调用
- `openai_compatible`: 通用OpenAI兼容API
- Factory模式: `ChatOpenAI` 实例工厂, 统一 timeout(60s) + max_retries(2) + temperature(0.2)

---

## 12. 数据库架构 (PostgreSQL)

### 12.1 核心表

| 表名 | 说明 | 关键字段 |
|------|------|---------|
| users | 用户 | id, email, hashed_password, nickname, status |
| user_profiles | 用户画像 | user_id, segment, goals(JSON), interests(JSON) |
| memories | 记忆 | user_id, memory_type(working/episodic/semantic/perceptual), content, importance(0-1), qdrant_point_id |
| feed_cards | Feed卡片 | user_id, info_item_id, title, one_sentence_value, information_gap, final_score, relation_type |
| info_items | 信息条目 | title, content, source_url, content_hash, source_type |
| documents | 文档 | user_id, filename, file_path, file_type, file_size, status |
| document_chunks | 文档分块 | document_id, chunk_index, content, qdrant_point_id, token_count |
| agent_runs | Agent运行 | user_id, conversation_id, thread_id, status, final_answer |
| agent_conversations | 会话 | user_id, conversation_id(uuid), title, thread_id |
| agent_chat_messages | 聊天消息 | conversation_id, role, content, metadata |
| agent_steps | 步骤记录 | run_id, node_name, input, output, status |
| agent_events | SSE事件 | run_id, event_type, node_name, payload |
| llm_calls | LLM调用日志 | run_id, purpose, provider, model, latency_ms, tokens |
| research_runs | 研究运行 | id(uuid), feed_card_id, query, status, findings, markdown_report |
| skills | 技能 | user_id, name, trigger_text, tool_plan, safety_level, status(draft/approved/disabled) |
| approvals | 审批记录 | pending/approved/rejected |
| artifacts | 成果物 | markdown文件路径+元数据 |
| mcp_servers / mcp_tools | MCP配置 | 服务器配置+工具注册 |
| feed_feedback | Feed反馈 | thumbs_up/down, dismissed |
| feed_refresh_attempts | Feed刷新记录 | 新增表 |
| eval_records | 评估记录 | 质量评估 |

---

## 13. 关键设计模式

### 13.1 Repository 模式
所有数据库操作通过 `db/repositories/` 封装, BaseRepository提供通用CRUD

### 13.2 Service 单例模式
业务逻辑Service在模块级别实例化: `xxx_service = XxxService()`

### 13.3 多层降级策略 (Fallback Chain)
- **意图识别**: LLM (`intent_llm.py`) → 规则引擎 (100+关键词)
- **LangGraph加载**: 正常图 → fallback顺序节点执行
- **记忆检索**: Qdrant语义搜索 → PostgreSQL ILIKE → 最近重要记忆
- **研究执行**: 上游ODR Adapter → FallbackResearcher
- **最终回答**: LLM流式生成 → 规则兜底文案
- **Feed**: 7个真实源 → BucketSeed → ManualSeed
- **记忆提取**: LlmMemoryExtractor (LLM) → MemoryExtractor (正则)
- **工具选择**: ToolRouter LLM引擎 → 关键词匹配

### 13.4 非阻塞设计
- RAG搜索失败不阻断Agent流水线
- 记忆提取失败不阻断Agent Run
- 图片分析失败返回友好错误信息
- Feed刷新单个源超时/失败不影响其他源

### 13.5 安全防护
- 权限守卫节点在所有处理之前执行
- L3/L4操作需要审批 (含 true graph interrupt → 等待 → resume 完整流程)
- final_response 防护内部JSON泄露
- JWT认证 (python-jose) + bcrypt密码哈希
- 会话并发锁 (conversation_lock.py)
- 本地文件操作沙箱隔离 (workspace目录 + 敏感文件阻断 + 路径穿越检测)
- SMTP密码不记录到日志

---

## 14. 上游开源项目 (src/open_deep_research/) 架构

### 14.1 主工作流

```
START → clarify_with_user ──(need_clarification?)──→ END (返回问题)
           │ (no)
           ▼
      write_research_brief (生成研究简报)
           │
           ▼
      research_supervisor (子图: supervisor ⇄ supervisor_tools)
           │                    ├─ think_tool (战略反思)
           │                    ├─ ConductResearch → researcher子图 (并行执行)
           │                    └─ ResearchComplete → 结束
           ▼
      final_report_generation (综合报告生成) → END
```

### 14.2 Researcher 子图 (每个并行实例)

```
START → researcher ⇄ researcher_tools → compress_research → END
```

- **researcher**: 使用搜索工具 + think_tool进行研究
- **researcher_tools**: 并行执行所有tool_call, 支持Tavily/OpenAI/Anthropic/MCP
- **compress_research**: 压缩研究成果, 去重+整理, 保留全部信息来源

### 14.3 Agent OS 适配配置 (覆盖上游默认值)

通过 Settings 中的 `odr_*` 前缀配置:
- `odr_research_model`: `openai:qwen-plus`
- `odr_allow_clarification`: `False`
- `odr_max_concurrent_research_units`: `2`
- `odr_max_researcher_iterations`: `2`
- `odr_max_react_tool_calls`: `4`
- `odr_timeout_seconds`: `600`

---

## 15. API 端点总览

| 前缀 | 关键端点 |
|------|---------|
| /api/v1/agent | `POST /runs/stream` (SSE), `POST /runs/{id}/resume` (审批恢复), CRUD conversations/messages |
| /api/v1/research | `POST /runs` (异步后台), `GET /runs/:id` |
| /api/v1/documents | `POST /upload`, `POST /ingest`, `POST /rag/search`, `POST /rag/ask`, `POST /documents/:id/ask` |
| /api/v1/memory | `POST /add`, `POST /search`, `POST /consolidate`, `POST /forget` |
| /api/v1/artifacts | CRUD + download |
| /api/v1/skills | CRUD + approve/disable |
| /api/v1/mcp | `GET /tools`, `POST /execute` |
| /api/v1/approvals | approve/reject |
| /api/v1/auth | login/register |
| /api/v1/profile | GET/PUT |
| /api/v1/feed | `GET /home`, `POST /refresh`, `POST /cards/:id/research` |
| /api/health | 健康检查 |

---

## 16. 测试覆盖

### 16.1 集成测试 (tests/)
| 文件 | 行数 | 覆盖范围 |
|------|------|---------|
| `test_approval_workflow.py` | 1148 | 完整审批工作流 (创建→暂停→审批→恢复→完成) |
| `test_llm_tool_selection.py` | 302 | LLM工具选择引擎 |
| `test_tool_input_validation.py` | 99 | 工具输入参数校验 |
| `test_tool_name_normalization.py` | 73 | 工具名规范化 + 中英文别名映射 |

### 16.2 单元测试 (src/web_app/tests/)
| 文件 | 行数 | 覆盖范围 |
|------|------|---------|
| `test_memory_system.py` | 560 | 记忆系统综合测试 (CRUD, 搜索, 固化, 遗忘, 提取, LLM/正则双引擎, 去重) |

---

## 17. 提交历史 (完整, 截至 2026-06-10)

```
08f5874 ← HEAD (feature/bendicaozuo): 下一步rdrant
422d7ec: 下一步，多智能体协作
85aa336: 下一步LLM识别
4ed9500: 完成了邮件的发送，下一步LLM接入识别意图
24cebd9: 下一步修思考过程中的出现审阅bug
31571e3: 完成了深度研究，下一步本地电脑操作
2406e69: 完成了信息差卡片基本情况，下一步操作电脑
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

## 18. 当前状态总结

### 已完成
- 基础Agent框架 (LangGraph 14+节点多智能体图 + true interrupt审批)
- RuntimeEventEmitter 统一事件发射器 (SSE推送+DB持久化一步完成)
- LLM意图识别 (intent_llm.py, 17种意图类型, LLM+规则双引擎)
- 会话管理 (多轮对话, agent_conversations/agent_chat_messages)
- Feed系统 (7+1源聚合, 评分, 混排30/40/30, 信息差卡片)
- 记忆系统 (三层架构 + Qdrant语义检索 + 双重提取器 + 向量回填 + Jaccard去重)
- RAG系统 (文档上传/解析/分块/摄入/检索问答/文档Q&A)
- 图片识别 (Qwen多模态, 直接问答快速通道, 混合附件处理)
- SSE流式输出 (4 channel: thinking/answer/tool/status)
- Skill系统 (匹配/创建/审批/自动草稿)
- MCP工具系统 (工具注册/LLM+关键词双引擎路由/审批/审计)
- 邮件工具 (Mock + SMTP, 中英文别名)
- 本地文件操作 (workspace沙箱, 敏感文件阻断, 路径穿越检测)
- 审批工作流 (完整ApprovalCard前端 + true graph interrupt + resume + 测试)
- LLM配置体系 (3 tier × 12 purpose, aliyun DashScope为主)
- 多层降级策略 (LLM→规则, Qdrant→PostgreSQL→recent, 上游→fallback)
- 深度研究 (上游ODR Adapter + ResearchService编排 + 前端ResearchRunDetailPage)
- 前端Markdown渲染 (react-markdown + remark-gfm)
- 会话并发锁 (conversation_lock.py)
- 前端ApprovalCard组件
- 全局CSS样式
- 测试覆盖 (5个测试文件, 共2182行)

### 当前开发焦点 (feature/bendicaozuo)
1. **Qdrant完善** — 向量存储优化, 索引完善
2. **多智能体协作** — Agent间协调机制
3. **LLM识别完善** — home_intent_react 节点的 LLM 判断器, 结构化意图输出
4. **本地电脑操作完善** — local_file_tools 功能增强

### 已知待优化
- 思考过程中出现的审阅bug
- LLM回答质量 (有时输出JSON而非自然语言)
- 流式输出体验优化
- 远域 (remote domain) 功能
- 文档相关bug修复

### 用户约束
- 不要修改 `src/open_deep_research/` 目录 (上游原始项目)
- 不重写Agent Runtime核心, 不破坏现有API
- 偏好中文表达, 简洁, 产品化风格
- 不要把内部JSON暴露给普通用户

---

## 19. 开发命令

```bash
# 环境准备
cp .env.example .env
# 编辑 .env 配置数据库连接、LLM API key等

# 启动服务
uvx langgraph dev                         # LangGraph Studio (上游)
uvicorn src.web_app.main:app --reload     # FastAPI Agent OS (主服务, 端口8000)

# 前端开发
cd frontend && npm run dev                # Vite React 开发服务器 (localhost:5173)

# 数据库迁移
alembic upgrade head

# Qdrant 索引
python scripts/ensure_qdrant_indexes.py
python scripts/backfill_memory_vectors.py

# 代码质量
ruff check                                # linting (E, F, I, D, UP规则)
mypy                                      # type checking

# 测试
pytest src/web_app/tests/                 # 单元测试 (记忆系统)
pytest tests/                             # 集成测试 (审批/工具)
python tests/run_evaluate.py              # LangSmith评估
```
