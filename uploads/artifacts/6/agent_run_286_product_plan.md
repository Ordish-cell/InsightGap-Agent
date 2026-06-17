# 研究摘要

Based on 5 evidence items, the research question '我的文档你没读到吗？？' points to a traceable information-gap opportunity. Key evidence: ### 10.3.1 你到底做了什么

你做的是一套“可评估的文档检索链路”，不是简单向量搜索。

它包括：

1. 文档解析。
2. 结构化 chunking。
3. child-only vector upsert。
4. Qdrant dense + sparse hybrid。
5. native RRF fusion。
6. parent context enrichment。
7. rerank。
8. hit@k eval runner。

把这几个词串起来，就是你的 RAG 故事。 #### 阶段一：用户请求进入服务层

用户在前端输入问题，比如：

```text
帮我总结一下当前上传的文档
```

或者：

```text
帮我把这段内容写入本地文件
```

请求进入后端 `AgentService`。服务层会做几件事：

1. 确定 `user_id`。
2. 创建或复用 `conversation_id`。
3. 创建本次执行的 `AgentRun`。
4. 生成或读取 `thread_id`。
5. 把 `user_input`、`page_context

## 关键发现


- {'title': 'Finding from rag', 'detail': '### 10.3.1 你到底做了什么\n\n你做的是一套“可评估的文档检索链路”，不是简单向量搜索。\n\n它包括：\n\n1. 文档解析。\n2. 结构化 chunking。\n3. child-only vector upsert。\n4. Qdrant dense + sparse hybrid。\n5. native RRF fusion。\n6. parent context enrichment。\n7. rerank。\n8. hit@k eval runner。\n\n把这几个词串起来，就是你的 RAG 故事。', 'confidence': 0.5, 'evidence_refs': ['agent面试.md']}

- {'title': 'Finding from rag', 'detail': '#### 阶段一：用户请求进入服务层\n\n用户在前端输入问题，比如：\n\n```text\n帮我总结一下当前上传的文档\n```\n\n或者：\n\n```text\n帮我把这段内容写入本地文件\n```\n\n请求进入后端 `AgentService`。服务层会做几件事：\n\n1. 确定 `user_id`。\n2. 创建或复用 `conversation_id`。\n3. 创建本次执行的 `AgentRun`。\n4. 生成或读取 `thread_id`。\n5. 把 `user_input`、`page_context`、`conversation_id` 等放进初始 state。\n\n这一步的意义是：**每一次 Agen', 'confidence': 0.5, 'evidence_refs': ['agent面试.md']}

- {'title': 'Finding from rag', 'detail': '#### 阶段二：LangGraph Runtime 接管执行\n\n服务层不会自己写一堆 if/else 把任务跑完，而是调用 `AgentRuntime.run()`。Runtime 内部使用 LangGraph `StateGraph`，共享状态是 `AgentRuntimeState`。\n\n初始 state 大概是：\n\n```python\n{\n    "user_id": 1,\n    "run_id": 123,\n    "thread_id": "user:1:conversation:abc",\n    "conversation_id": "abc",\n    "user_inpu', 'confidence': 0.2, 'evidence_refs': ['agent面试.md']}

- {'title': 'Finding from rag', 'detail': '5. **parallel_prefetch / parallel_read_stage 提前准备上下文**\n   这里会提前准备 Memory、RAG evidence、Skill candidates、conversation history、FeedCard context 等。这样后面的 RAG、final_response、skill_matcher 不用各自重复查。\n\n6. **supervisor_observer → llm_supervisor_route → dispatcher 做动态节点选择**\n   你的项目里没有一个单独名叫 router 的 StateGraph ', 'confidence': 0.16666667, 'evidence_refs': ['agent面试.md']}

- {'title': 'Finding from rag', 'detail': '### 10.3.8 RAG 模块长版面试讲法\n\n> RAG 模块我主要做了两件事：一是改 chunking，二是改检索融合，并且补了评估。\n>\n> chunking 上，我没有把文档直接等长切片，而是做 Parent-Child。解析文档后生成 overview、parent、child 三类 chunk。只有 child chunk 会 embedding 并写入 Qdrant，因为 child 粒度小，召回更准；parent 和 overview 保存在 PostgreSQL。检索命中 child 后，再根据 parent_id 回查 parent_context，给最终回答补充完整上下', 'confidence': 0.25, 'evidence_refs': ['agent面试.md']}

# 知识库检索

以下是从当前上传文档中检索到的相关内容，请基于这些内容回答：

