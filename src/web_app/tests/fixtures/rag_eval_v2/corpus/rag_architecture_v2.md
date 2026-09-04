# RAG Architecture V2（当前生产方案）

状态：active；版本 2.3。

## 检索链路

当前方案组合 dense embedding 与 sparse lexical retrieval，在 Qdrant 中使用 RRF 融合排序。主配置键是 `RAG_HYBRID_BACKEND=qdrant_hybrid`，集合名称为 `rag_hybrid_v2`，RRF 参数 k=60。

## Parent-Child Chunking

Parent 目标长度为 800 tokens；Child 目标长度为 180 tokens，overlap 为 40 tokens。Child 用于精确召回，命中后通过 parent_id 回查 Parent 作为回答上下文。

## 降级策略

Qdrant 不可用时回退到 `python_bm25`。回退必须产生 `qdrant_hybrid_failed` warning，不允许静默伪装成 Hybrid 成功。

## 质量门禁

冻结测试集要求 Evidence Recall@5 不低于 0.90，MRR@10 不低于 0.75，并单独报告 P95 检索延迟。
