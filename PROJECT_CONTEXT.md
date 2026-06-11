# Open Deep Research - Agent OS 项目上下文

> 最后更新：2026-06-11  
> 本文用于后续开发、调试、面试讲解和 AI 协作快速建立项目全局认知。  
> 不记录 `.env` 中的真实密钥。

## 项目定位

本项目基于 `langchain-ai/open_deep_research` 二次开发，二开核心位于 `src/web_app/`，上游原始目录 `src/open_deep_research/` 原则上不改。

核心闭环：

```text
Feed 信息接入 -> Agent 编排与研究 -> Artifact 成果沉淀 -> Memory / Skill 长期沉淀
```

主要技术栈：

| 模块 | 技术 |
|---|---|
| Web API | FastAPI |
| Agent Runtime | LangGraph + 自研 RuntimeNodes |
| LLM | DashScope / Qwen，OpenAI-compatible 接入 |
| 文档 RAG | PostgreSQL + Qdrant Cloud |
| 长期记忆 | PostgreSQL + Qdrant `memory_vectors` |
| 缓存 | Redis |
| 前端 | Vite + React + TypeScript |

## 当前 RAG 配置

当前主文档检索后端是 Qdrant Native Hybrid：

```text
RAG_HYBRID_BACKEND=qdrant_hybrid
QDRANT_HYBRID_COLLECTION=agent_os_documents_v3
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=bm25
QDRANT_SPARSE_ENCODER=qdrant_cloud_bm25
QDRANT_SPARSE_MODEL=Qdrant/bm25
QDRANT_CLOUD_INFERENCE=true
QDRANT_FUSION_METHOD=rrf
QDRANT_HYBRID_FALLBACK=true
```

含义：

- `agent_os_documents_v3` 是官方 Qdrant BM25 sparse input collection。
- sparse input 使用 `models.Document(text=..., model="Qdrant/bm25")`。
- v3 不混写旧 hashing sparse。
- Python BM25 保留为生产 fallback。
- 旧 v2 hashing sparse 只用于 eval / 手动对比，不自动混入生产请求。

Embedding：

```text
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v4
QDRANT_VECTOR_SIZE=1024
DASHSCOPE_EMBEDDING_BATCH_SIZE=10
```

## 文档入库链路

```text
上传文件
  -> 保存到 storage/uploads/{user_id}/{document_id}/
  -> parse_document()
  -> build_structured_chunks()
  -> 生成 overview / parent / child
  -> 只对 child 做 embedding
  -> child 写 Qdrant
  -> overview / parent / child 写 PostgreSQL
  -> documents.metadata_json 写 overview / document_map / chunking_stats
```

chunk 语义：

| 类型 | 作用 | PG | Qdrant |
|---|---|---:|---:|
| overview | 总结类问题入口 | yes | no |
| parent | 回答上下文 | yes | no |
| child | 检索命中 | yes | yes |

`chunk_count` 只表示可检索 child chunk 数。

## 当前检索链路

主链路：

```text
query
  -> DashScope dense embedding
  -> Qdrant dense prefetch
  -> Qdrant BM25 sparse prefetch via models.Document
  -> Qdrant RRF fusion
  -> child hits
  -> PG parent lookup
  -> parent_context enrich
  -> RAGService evidence
```

fallback：

```text
qdrant_cloud_bm25 search 失败
  -> Python BM25 fallback
  -> 保留 retrieval_warning
```

ingest/upsert 约束：

- cloud BM25 upsert 失败时，文档摄入失败。
- 不在 `agent_os_documents_v3` 中 fallback 写 hashing sparse。
- parent / overview 不入 Qdrant。
- 删除文档向量只按 `user_id + document_id` 删除 document collection，不影响 `memory_vectors`。

## 关键文件

```text
src/web_app/services/document_service.py
src/web_app/services/rag_service.py
src/web_app/rag/vector_store.py
src/web_app/rag/sparse_encoder.py
src/web_app/rag/retriever.py
src/web_app/rag/bm25.py
src/web_app/rag/structured_chunker.py
src/web_app/db/repositories/document_repository.py
```

## 最新 synthetic eval

报告：

```text
uploads/artifacts/rag_eval/rag_hybrid_eval_20260611_145414.md
```

指标：

| backend | hit@1 | hit@3 | hit@5 | keyword_hit_rate | fallback | warnings | avg_latency_ms | usable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| python_bm25 | 0.46 | 0.54 | 0.54 | 0.54 | 0 | 0 | 6312.37 | no |
| qdrant_hybrid | 0.54 | 0.69 | 0.92 | 0.92 | 0 | 0 | 7867.48 | yes |

结论：

```text
qdrant_hybrid 已在 synthetic eval 中跑通 agent_os_documents_v3 + Qdrant/bm25。
Top-5 与关键词命中率均为 0.92。
生产全量切换前仍建议保留 Python BM25 fallback，并继续做真实文档小批量验证。
```
