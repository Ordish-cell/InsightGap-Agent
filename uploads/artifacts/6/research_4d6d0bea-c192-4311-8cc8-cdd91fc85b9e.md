# Research Report: 我的文档你没读到吗？？

## 1. Executive Summary

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

## 2. Why This Matters

This research expands a FeedCard information gap into a traceable report grounded in available evidence.

## 3. Key Findings

- **Finding from rag**: ### 10.3.1 你到底做了什么

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

把这几个词串起来，就是你的 RAG 故事。
- **Finding from rag**: #### 阶段一：用户请求进入服务层

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
5. 把 `user_input`、`page_context`、`conversation_id` 等放进初始 state。

这一步的意义是：**每一次 Agen
- **Finding from rag**: #### 阶段二：LangGraph Runtime 接管执行

服务层不会自己写一堆 if/else 把任务跑完，而是调用 `AgentRuntime.run()`。Runtime 内部使用 LangGraph `StateGraph`，共享状态是 `AgentRuntimeState`。

初始 state 大概是：

```python
{
    "user_id": 1,
    "run_id": 123,
    "thread_id": "user:1:conversation:abc",
    "conversation_id": "abc",
    "user_inpu
- **Finding from rag**: 5. **parallel_prefetch / parallel_read_stage 提前准备上下文**
   这里会提前准备 Memory、RAG evidence、Skill candidates、conversation history、FeedCard context 等。这样后面的 RAG、final_response、skill_matcher 不用各自重复查。

6. **supervisor_observer → llm_supervisor_route → dispatcher 做动态节点选择**
   你的项目里没有一个单独名叫 router 的 StateGraph 
- **Finding from rag**: ### 10.3.8 RAG 模块长版面试讲法

> RAG 模块我主要做了两件事：一是改 chunking，二是改检索融合，并且补了评估。
>
> chunking 上，我没有把文档直接等长切片，而是做 Parent-Child。解析文档后生成 overview、parent、child 三类 chunk。只有 child chunk 会 embedding 并写入 Qdrant，因为 child 粒度小，召回更准；parent 和 overview 保存在 PostgreSQL。检索命中 child 后，再根据 parent_id 回查 parent_context，给最终回答补充完整上下

## 4. Evidence

- **agent面试.md** (rag, score=0.5): ### 10.3.1 你到底做了什么

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

把这几个词串起来，就是你的 RAG 故事。
- **agent面试.md** (rag, score=0.5): #### 阶段一：用户请求进入服务层

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
5. 把 `user_input`、`page_context`、`conversation_id` 等放进初始 state。

这一步的意义是：**每一次 Agent 执行都有数据库身份**。后续的节点事件、工具调用、审批记录、最终答案都能挂到同一个 run 上。
- **agent面试.md** (rag, score=0.2): #### 阶段二：LangGraph Runtime 接管执行

服务层不会自己写一堆 if/else 把任务跑完，而是调用 `AgentRuntime.run()`。Runtime 内部使用 LangGraph `StateGraph`，共享状态是 `AgentRuntimeState`。

初始 state 大概是：

```python
{
    "user_id": 1,
    "run_id": 123,
    "thread_id": "user:1:conversation:abc",
    "conversation_id": "abc",
    "user_input": "帮我总结一下当前上传的文档",
    "page_context": {...},
    "mode": "react"
}
```

LangGraph 的作用是把这份 state 依次传给不同节点，每个节点读取一部分、写回一部分。
- **agent面试.md** (rag, score=0.16666667): 5. **parallel_prefetch / parallel_read_stage 提前准备上下文**
   这里会提前准备 Memory、RAG evidence、Skill candidates、conversation history、FeedCard context 等。这样后面的 RAG、final_response、skill_matcher 不用各自重复查。

6. **supervisor_observer → llm_supervisor_route → dispatcher 做动态节点选择**
   你的项目里没有一个单独名叫 router 的 StateGraph 节点。实际路由分成：

| 层 | 责任 |
   |---|---|
   | planner | 生成初始 route plan |
   | supervisor_observer | 观察当前执行状态和候选节点 |
   | llm_supervisor_route | LLM 分析 state，可选改写 route plan（full 模式） |
   | dispatch_next_r
- **agent面试.md** (rag, score=0.25): ### 10.3.8 RAG 模块长版面试讲法

> RAG 模块我主要做了两件事：一是改 chunking，二是改检索融合，并且补了评估。
>
> chunking 上，我没有把文档直接等长切片，而是做 Parent-Child。解析文档后生成 overview、parent、child 三类 chunk。只有 child chunk 会 embedding 并写入 Qdrant，因为 child 粒度小，召回更准；parent 和 overview 保存在 PostgreSQL。检索命中 child 后，再根据 parent_id 回查 parent_context，给最终回答补充完整上下文。
>
> 检索上，我把 Qdrant 从 dense-only 升级成 dense+sparse hybrid collection。写入时 child 同时有 dense embedding 和 sparse vector；查询时 query 也同时生成 dense 和 sparse，两路在 Qdrant prefetch，然后用 Fusion.RRF 做融合。这样能同时覆盖语义问题和合同编

## 5. Information Gap Analysis

The key information gap is whether the observed signal is merely a news item or an actionable opportunity for the user.

## 6. Opportunities

- **Generate a focused report**: Convert the FeedCard signal into a concise research artifact for later comparison.
- **Create a reusable skill draft**: Capture the repeated workflow: load signal, gather evidence, build GSSC context, produce report.
- **Follow the strongest source**: Start from: agent面试.md

## 7. Risks and Uncertainties

- **Low confidence source**: At least one source has low credibility or weak retrieval score.

## 8. Suggested Actions

- **Save report**: Keep the markdown artifact for later review.
- **Add memory**: Record this research run as episodic memory.
- **Review evidence**: Open sources manually before making product or investment decisions.

## 9. Sources

- agent面试.md: no URL
- agent面试.md: no URL
- agent面试.md: no URL
- agent面试.md: no URL
- agent面试.md: no URL