# RAG Eval V2 Dataset

这是一套不依赖 PostgreSQL、Qdrant、Embedding 或 LLM 即可生成和校验的合成 Gold Set。

## 内容

- `corpus/`：22 份 Markdown、TXT/CSV 测试文档，包含版本冲突、相似字段和 hard negatives。
- `cases.dev.jsonl`：70 条开发集，可用于调参。
- `cases.test.jsonl`：30 条冻结测试集，只用于最终报告，不应针对它反复调参。
- `manifest.json`：题型数量、split 数量和语料 SHA-256。

题型共 100 条：

| category | count |
|---|---:|
| exact | 20 |
| table | 15 |
| semantic | 20 |
| parent_context | 15 |
| conflict | 10 |
| summary | 10 |
| unanswerable | 10 |

## 重新生成

```powershell
.venv\Scripts\python.exe scripts\build_rag_eval_v2_dataset.py
```

## 离线校验

```powershell
.venv\Scripts\python.exe -m pytest src\web_app\tests\test_rag_eval_v2_dataset.py -q
```

## Gold Case 结构

- `relevant_documents`：正确证据所在文档。
- `gold_evidence`：稳定的 filename、heading 和 required facts，不依赖易变化的 chunk id。
- `hard_negative_documents`：内容相似但不应被当成最终证据的文档。
- `answerable=false`：系统应拒答，不能根据相似片段编造。

Qdrant 开启后，应把该数据集接入真实 `DocumentService -> StructuredChunker -> Qdrant -> ParentChildRetriever` 链路，再分别计算 Document/Evidence Recall、MRR、nDCG、拒答率和 P95 延迟。
