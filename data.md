# Open Deep Research - Agent OS 项目上下文

> 最后更新：2026-06-11  
> 用途：给后续开发、调试、面试讲解和 AI 协作快速建立项目全局认知。  
> 注意：本文只记录架构和配置形态，不记录 `.env` 中的真实密钥。

---

## 1. 项目定位

本项目基于开源项目 `langchain-ai/open_deep_research` 二次开发，目标是构建一个面向个人信息工作流的 Agent OS 平台。

核心闭环：

```text
Feed 信息接入
  -> Agent 编排与研究
  -> Artifact 成果沉淀
  -> Memory / Skill 长期沉淀
  -> 再反哺 Agent 个性化处理
```

当前后端主栈：

| 模块 | 技术 |
|---|---|
| Web API | FastAPI |
| Agent Runtime | LangGraph + 自研 RuntimeNodes |
| LLM | DashScope / Qwen 为主，OpenAI-compatible 方式接入 |
| RAG 文档库 | PostgreSQL + Qdrant Cloud |
| 长期记忆 | PostgreSQL + Qdrant `memory_vectors` |
| 缓存 / 工作记忆 | Redis |
| 数据库 | PostgreSQL |
| 前端 | Vite + React + TypeScript |

开发约束：

- 不直接修改 `src/open_deep_research/` 上游原始目录。
- 二开能力集中在 `src/web_app/`。
- 不破坏现有 API。
- Agent Runtime 不做大面积重写，优先局部增强。
- 文档 RAG 与 memory vectors 生命周期必须隔离。

---

## 2. 当前运行配置摘要

当前 `.env` 中的 RAG 检索后端已经切到 Qdrant Native Hybrid：

```text
RAG_HYBRID_BACKEND=qdrant_hybrid
QDRANT_HYBRID_COLLECTION=agent_os_documents_v2
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=bm25
QDRANT_FUSION_METHOD=rrf
QDRANT_HYBRID_FALLBACK=true
QDRANT_SPARSE_HASH_SIZE=2000003
```

Qdrant 基础配置：

```text
QDRANT_COLLECTION=hello_agents_vectors
QDRANT_VECTOR_SIZE=1024
QDRANT_DISTANCE=cosine
QDRANT_TIMEOUT=30
```

解释：

- `hello_agents_vectors` 是普通 dense collection 配置。
- 当前主文档 RAG 因为 `RAG_HYBRID_BACKEND=qdrant_hybrid`，实际使用 `agent_os_documents_v2`。
- `agent_os_documents_v2` 需要同时支持 named dense vector `dense` 和 sparse vector `bm25`。
- Python BM25 没有删除，当前作为 fallback。
- Qdrant hybrid 失败时，只要 `QDRANT_HYBRID_FALLBACK=true`，会回退到 Python BM25 hybrid fallback。

Embedding 配置：

```text
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v4
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_EMBEDDING_BATCH_SIZE=10
```

说明：

- 当前 embedding 模型是 DashScope `text-embedding-v4`。
- 当前向量维度配置为 1024。
- DashScope 单批 input 最大 10 条，代码已按 `DASHSCOPE_EMBEDDING_BATCH_SIZE=10` 分批请求。
- 如果配置超过 10，会自动 clamp 到 10 并记录 warning。

---

## 3. 关键目录

```text
src/
  open_deep_research/        # 上游原始目录，原则上不改
  web_app/                   # Agent OS 二开核心
    api/v1/                  # FastAPI 路由
    services/                # 业务服务层
    rag/                     # RAG / 文档解析 / 检索
    memory/                  # 长期记忆系统
    agent/runtime/           # LangGraph Agent Runtime
    context/                 # GSSC 上下文构建
    db/repositories/         # Repository 数据访问层
    models/                  # SQLAlchemy ORM
    tests/                   # 后端单元测试
scripts/                     # 运维、评估和验证脚本
uploads/artifacts/rag_eval/  # RAG eval 输出报告
storage/uploads/             # 用户上传文件存储
```

RAG 重点文件：

```text
src/web_app/services/document_service.py
src/web_app/services/rag_service.py
src/web_app/db/repositories/document_repository.py
src/web_app/rag/document_parser.py
src/web_app/rag/structured_chunker.py
src/web_app/rag/embeddings.py
src/web_app/rag/vector_store.py
src/web_app/rag/retriever.py
src/web_app/rag/bm25.py
src/web_app/rag/query_analyzer.py
src/web_app/rag/reranker.py
src/web_app/rag/sparse_encoder.py
```

---

## 4. 文档上传与入库链路

普通文档上传：

```text
POST /api/v1/documents/upload
  -> DocumentService.upload_document()
  -> 校验文件名 / 扩展名 / 大小
  -> 写 documents，status=uploaded
  -> 文件保存到 storage/uploads/{user_id}/{document_id}/
  -> 返回 document 信息

POST /api/v1/documents/{document_id}/ingest
  -> DocumentService.ingest_document()
  -> 同步执行解析、分块、embedding、向量入库、PG chunks 写入
```

聊天附件上传：

```text
POST /api/v1/documents/chat-upload
  -> DocumentService.upload_chat_attachment()
  -> kind=image 时跳过 RAG 入库
  -> kind=document 时同步调用 _ingest_document_internal()
  -> 成功返回 ingest_status=ingested
  -> 失败返回 DOCUMENT_INGEST_FAILED，并带 document_id / ingest_status / error_message
```

状态流转：

```text
uploaded
  -> ingesting
      failed_stage=qdrant_delete
      failed_stage=parse
      failed_stage=chunk
      failed_stage=embed
      failed_stage=qdrant_upsert
      failed_stage=db_write
  -> ingested
```

失败状态：

```text
document.status=failed
metadata_json.ingest_status=failed
metadata_json.failed_stage=...
metadata_json.error=...
metadata_json.error_message=...
```

历史兼容状态：

```text
ingested
completed
ready
```

---

## 5. 文档解析链路

解析入口：

```text
src/web_app/rag/document_parser.py
parse_document()
```

支持类型：

```text
txt
md
pdf
docx
xlsx
csv
json
html / htm
```

解析策略：

```text
优先 markitdown
  -> 成功：parser=markitdown, used_fallback=false
  -> 失败：记录 fallback_reason
  -> 进入专用 fallback parser
```

Fallback parser：

| 类型 | fallback |
|---|---|
| txt / md | 直接读文本，utf-8 / gbk / ignore |
| pdf | PyMuPDF |
| docx | python-docx |
| xlsx | openpyxl |
| csv | csv.reader |
| json | json.loads + pretty JSON |
| html | BeautifulSoup get_text |

PDF 特殊处理：

- 如果 markitdown 成功，也会尝试用 PyMuPDF 补充 `page_count` 和 `pages`。
- 如果页码可取，后续 structured chunker 可以按页生成 parent。
- 页码拿不到时不引入重型依赖，留空。

---

## 6. 分块设计：overview / parent / child

分块入口：

```text
src/web_app/rag/structured_chunker.py
build_structured_chunks()
```

当前入库产物分为三类：

| 类型 | 作用 | 是否写 PG | 是否写 Qdrant |
|---|---|---:|---:|
| overview | 文档摘要和结构概览 | yes | no |
| parent | 回答上下文块 | yes | no |
| child | 检索命中块 | yes | yes |

设计原则：

```text
child 负责检索命中
parent 负责回答上下文
overview / document_map 负责总结类问题
citation 仍指向 child
```

`chunk_count` 语义：

```text
chunk_count = 可检索 child chunk 数
```

它不包含 parent 和 overview。

文本类文档分块：

```text
Markdown/text
  -> 按 heading / paragraph / code block 生成 parent
  -> parent 大致控制在 1400 tokens 内
  -> parent 再切成 child
  -> child_chunk_size=800
  -> child_overlap=80
```

PDF 分块：

```text
如果 parser_metadata.pages 可用：
  -> 每页一个 parent
  -> page_number 写入 metadata
否则：
  -> 退回普通文本结构化分块
```

CSV / XLSX 分块：

```text
按 sheet/header/row block 切
csv 默认每 40 行一个 row_block
xlsx 默认每 30 行一个 row_block
row_block 直接作为 child，不强套普通文本父子切分
```

关键 metadata：

```text
chunk_role
chunk_type
chunk_id
parent_id
heading_path
page_number
sheet_name
header
row_start
row_end
content_hash
```

---

## 7. Embedding 链路

入口：

```text
src/web_app/rag/embeddings.py
embed_texts()
```

当前配置：

```text
provider=dashscope
model=text-embedding-v4
dimension=1024
batch_size=10
```

校验规则：

```text
None -> 过滤
空字符串 -> 过滤
非 str -> 报错
超长 chunk -> DocumentService 提前拦截
embed_texts([]) -> []
```

DashScope batching：

```text
每批最多 10 条 input
多批结果按原 input 顺序拼接
任一批失败不吞异常
返回向量数必须等于输入文本数
```

失败日志会包含：

```text
batch_index
batch_start
batch_end
batch_size
status_code
response_text
model
length_stats
safe previews
```

---

## 8. Qdrant 文档向量存储

入口：

```text
src/web_app/rag/vector_store.py
QdrantVectorStore
```

当前主后端：

```text
RAG_HYBRID_BACKEND=qdrant_hybrid
collection=agent_os_documents_v2
dense vector name=dense
sparse vector name=bm25
fusion=RRF
```

Collection schema：

```text
vectors_config:
  dense:
    size: 1024
    distance: cosine

sparse_vectors_config:
  bm25:
    modifier: IDF
```

写入策略：

```text
只写 child chunks
parent / overview 不写 Qdrant
child point 同时包含 dense vector 和 sparse vector
payload 保留完整 citation / parent lookup metadata
```

核心 payload：

```text
user_id
document_id
chunk_id
qdrant_point_id
chunk_index
content
content_preview
filename
file_type
source_title
heading_path
token_count
content_hash
chunk_role
chunk_type
parent_id
page_number
sheet_name
header
row_start
row_end
created_at
metadata
```

删除策略：

```text
delete_document(user_id, document_id)
  -> Qdrant filter 同时包含 user_id 和 document_id
  -> 只删除文档向量 collection
  -> 不影响 memory_vectors
```

---

## 9. 当前检索策略

入口：

```text
POST /api/v1/rag/search
RAGService.search()
ParentChildRetriever.search()
```

当前 `.env` 主链路：

```text
query
  -> embed_text(query)
  -> QdrantVectorStore.search_hybrid()
  -> dense prefetch
  -> sparse BM25 prefetch
  -> Qdrant RRF fusion
  -> child hits
  -> PG parent lookup
  -> parent_context enrich
  -> 返回兼容 search results
```

Fallback 链路：

```text
如果 qdrant_hybrid 不支持 / 查询失败 / collection schema 不匹配：
  -> Python BM25 hybrid fallback
```

Python BM25 fallback：

```text
Qdrant dense vector search
  + PG child chunks Python BM25
  + 应用层 merge
  + 轻量 rerank
  + parent_context enrich
```

注意：

- 当前不是“只用 Python BM25”。
- 当前主检索后端以 `.env` 为准，是 `qdrant_hybrid`。
- Python BM25 仍保留，用于 fallback 和对比评估。

---

## 10. 父子检索闭环

Qdrant / BM25 命中的是 child：

```text
child_chunk_id = p-0003-c-001
parent_id = p-0003
```

然后通过 PG 回取 parent：

```text
DocumentChunkRepository.list_parent_chunks(
  user_id,
  {document_id: [parent_id]}
)
```

安全限制：

```text
必须带 user_id
必须带 document_id
parent row 必须 metadata_json.chunk_role=parent
```

返回结果增强字段：

```text
child_chunk_id
parent_id
parent_chunk_id
parent_db_chunk_id
parent_chunk_index
parent_context
parent_context_available
citation
```

回答上下文：

```text
有 parent -> evidence 使用 parent_context
无 parent -> fallback 使用 child content
citation 始终指向 child
```

---

## 11. 文档问答链路

普通 RAG 问答：

```text
POST /api/v1/rag/ask
  -> RAGService.ask()
  -> search()
  -> evidence_from_results()
  -> parent_context 优先
  -> extractive fallback answer
```

文档级问答：

```text
RAGService.ask_document()
```

总结类问题识别：

```text
总结
概括
讲什么
主要内容
overview
summary
summarize
what is this document
what is this file
```

总结类问题流程：

```text
document_ids 存在
  -> 读取 documents.metadata_json.overview
  -> 读取 documents.metadata_json.document_map
  -> 生成 overview evidence
  -> 再补充少量 parent/child retrieval evidence
```

Agent Runtime 接入：

```text
RuntimeNodes.rag_agent()
  -> document_qa + summary -> rag_service.ask_document()
  -> 其他 RAG 问题 -> rag_service.ask()
  -> rag_result 写入 state
  -> final_response 基于 evidence / document_context_block 生成最终自然语言回答
```

重要现状：

- `/rag/ask` 自身目前偏 extractive fallback，不是完整 answer contract。
- `ask_document()` 可能返回 `[document_qa_context]`，表示已准备文档上下文，最终答案由 Agent Runtime final_response 生成。
- final_response 有规则禁止输出内部 JSON、payload、chunk 等内部术语。

---

## 12. 当前 RAG 优化结论

已经完成的 RAG 优化：

1. 结构化入库：overview / parent / child。
2. Qdrant 只入 child，PG 保存 overview / parent / child。
3. parent-child retrieval 闭环。
4. Python BM25 fallback。
5. Qdrant Native Hybrid Retrieval：dense + sparse BM25 + RRF。
6. Synthetic RAG eval 自动验证脚本。
7. DashScope embedding batching 和错误可观测性。

当前推荐运行方式：

```text
staging / 本地验证：
  RAG_HYBRID_BACKEND=qdrant_hybrid
  QDRANT_HYBRID_COLLECTION=agent_os_documents_v2
  QDRANT_HYBRID_FALLBACK=true

生产全量切换前：
  继续保留 Python BM25 fallback
  对真实文档做小批量验证
  确认旧文档是否需要 reingest / backfill sparse vectors
```

主要风险：

- 老 collection 或老文档如果没有 sparse vector，需要重新摄入或 backfill。
- Qdrant Cloud 网络异常会影响主检索，需保留 fallback。
- 当前 synthetic eval 不能直接等同于线上真实召回率。
- 中文 query analyzer 仍有优化空间。
- `/rag/ask` 仍偏抽取式，最终自然语言回答主要依赖 Agent final_response。

---

## 13. 开发与验证命令

启动后端：

```bash
uvicorn src.web_app.main:app --reload
```

启动前端：

```bash
cd frontend
npm run dev
```

RAG hybrid 自动评估：

```bash
python scripts/run_rag_hybrid_eval.py
```

对比已有 fixtures：

```bash
python scripts/compare_rag_backends.py
```

关键测试：

```bash
python -m pytest -q src/web_app/tests/test_rag_stage3.py
python -m pytest -q src/web_app/tests/test_rag_qdrant_hybrid.py
python -m pytest -q src/web_app/tests/test_rag_hybrid_retrieval.py
python -m pytest -q src/web_app/tests/test_rag_hybrid_eval_runner.py
python -m pytest -q src/web_app/tests/test_embedding_observability.py
```

---

## 14. 量化数据与评测结果

> 本节刻意放在文档最后，方便对外讲解和简历复用时引用。  
> 数据来源：`uploads/artifacts/rag_eval/rag_hybrid_eval_20260611_105403.md`。  
> 评测方式：synthetic RAG eval，非真实线上用户 query。

### 14.1 评测文档

```text
chinese_risk_notes.md      document_id=25 status=ingested
contract_pdf_like.txt      document_id=26 status=ingested
product_inventory.csv      document_id=27 status=ingested
tech_agent_config.md       document_id=28 status=ingested
```

覆盖场景：

- Markdown 技术文档：函数名、配置 key、API endpoint、风险说明。
- 合同类文本：合同编号、金额、日期、邮箱、付款条款、退款条款。
- CSV 表格：产品型号、价格、数量、负责人、邮箱、状态字段。
- 中文说明文档：中文关键词、中文风险段落、混合检索说明。

### 14.2 Backend 对比

| backend | hit@1 | hit@3 | hit@5 | keyword_hit_rate | fallback | warnings | avg_latency_ms | usable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| python_bm25 | 0.46 | 0.54 | 0.54 | 0.54 | 0 | 0 | 6009.10 | no |
| qdrant_hybrid | 0.54 | 0.69 | 0.92 | 0.92 | 0 | 0 | 7615.74 | yes |

### 14.3 提升幅度

Hit@1：

```text
python_bm25   = 0.46
qdrant_hybrid = 0.54
绝对提升 = 0.08
相对提升 = 17.4%
```

Hit@3：

```text
python_bm25   = 0.54
qdrant_hybrid = 0.69
绝对提升 = 0.15
相对提升 = 27.8%
```

Hit@5：

```text
python_bm25   = 0.54
qdrant_hybrid = 0.92
绝对提升 = 0.38
相对提升 = 70.4%
```

Keyword hit rate：

```text
python_bm25   = 0.54
qdrant_hybrid = 0.92
绝对提升 = 0.38
相对提升 = 70.4%
```

Latency：

```text
python_bm25 avg_latency_ms   = 6009.10
qdrant_hybrid avg_latency_ms = 7615.74
增加 = 1606.64ms
相对增加 = 26.7%
```

### 14.4 结论

在 synthetic RAG eval 上，Qdrant Hybrid 相比 Python BM25 fallback 表现更好：

- Top-5 证据召回从 0.54 提升到 0.92。
- 关键词命中率从 0.54 提升到 0.92。
- fallback_count 和 warning_count 均为 0。
- 平均延迟增加约 26.7%，属于需要继续优化但可以接受的代价。

当前判断：

```text
qdrant_hybrid 已具备作为 staging 默认检索后端的条件。
生产全量默认切换前，仍建议先做真实文档小批量验证，并保留 python_bm25 fallback。
```

### 14.5 简历 / 面试表述

简洁版：

```text
负责 Agent OS 文档 RAG 检索链路优化，将自研 Python BM25 fallback 升级为 Qdrant dense+sparse hybrid retrieval，并使用 RRF 融合排序；在自建 synthetic RAG eval 基准集上，hit@5 从 0.54 提升至 0.92，keyword hit rate 从 0.54 提升至 0.92，相对提升约 70.4%，同时保留 Python BM25 fallback 保障稳定性。
```

工程版：

```text
设计并实现 RAG 检索后端评测体系，构造 Markdown 技术文档、合同文本、CSV 表格和中文说明等 synthetic fixtures，覆盖总结、精确字段、表格查询、技术配置和中文风险场景；基于离线评测对比 Python BM25 与 Qdrant dense+sparse hybrid retrieval，验证 Qdrant Hybrid 将 hit@5 与关键词命中率从 54% 提升至 92%，并保持 fallback_count=0、warning_count=0。
```

性能权衡版：

```text
对 RAG 检索后端进行量化选型，将 Python BM25 baseline 与 Qdrant dense+sparse + RRF hybrid retrieval 进行离线评测；在平均延迟增加约 26.7% 的情况下，将 Top-5 证据召回率和关键词命中率均从 54% 提升至 92%，据此将 Qdrant Hybrid 作为 staging 默认候选，同时保留 Python BM25 fallback 降低上线风险。
```
