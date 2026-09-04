# RAG Architecture V1（历史方案）

状态：deprecated。

## 检索

V1 仅使用 Python BM25，配置键为 `RAG_BACKEND=python_bm25`，索引名称为 `rag_chunks_v1`。

## 切分

所有文档按 500 tokens 固定长度切分，不保存 parent 层级，overlap 为 50 tokens。

## 已知问题

同义改写召回较弱，长段落被切断后缺少回答上下文。
