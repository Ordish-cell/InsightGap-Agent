# Gap 统一身份与行为设定

基础设定集中在 `src/web_app/agent/prompts.py` 的 `GAP_BASE_PROMPT`。修改这一处即可调整后续用户回答的身份、语气及事实边界；`gap_system_message()` 是纯组合函数，无配置查询或网络调用。

## 接入范围

- 普通聊天和澄清：与原首行路由协议合并为一个系统消息，文件指代协议保持原状。
- 文档回答：与文件引用、覆盖范围和证据不足规则合并。
- 工作流最终回答：GSSC、旧上下文、会话回顾统一在调用处加入系统消息；现有任务内容和插话补充作为任务消息保留。移除了“你是最终回复节点”等冲突身份措辞。
- 图片直接回答：增加统一系统消息；内部图片分析不注入。
- 内部意图识别、Supervisor、摘要、研究调度及固定错误提示不变。不在模型工厂全局注入。

不新增节点或模型调用，不修改 REST/SSE、数据库、模型配置、历史消息、取消机制及上下文选择算法。普通聊天仍一次模型调用，正常文档回答仍两次。

基础设定为 394 字符，使用 cl100k 编码估算为 382 token；任务协议组合还会有少量连接文本。不同供应商实际分词可能不同。现有上下文预算维持原逻辑，没有把设定重复放入历史或检索资料。工作流调用账本的输入字符统计已计入系统消息；此改动不新增完整请求的上下文上限管理。

## 验证记录：2026-09-10

Windows、本项目虚拟环境、临时 SQLite 会话。

- 相关回归：**133 passed、7 skipped**。覆盖身份、聊天短路径、文档跨轮问答、停止／插话、运行兼容、供应商配置、后台摘要。跳过包括 6 个显式开启的真实身份冒烟及 1 个既有文档冒烟。
- 独立运行真实身份冒烟：**6 passed**，沿用现有 `qwen3.7-flash` 配置。只读获取模型配置，测试消息、文件内容及账本保存在临时数据库；未调用写工具，未写入用户会话。
- “你是谁”明确回答 Gap 信息差助手；疲惫倾诉、RAG 解释未重复品牌自我介绍；文档回答使用合成实验报告并标注文件名；工作流最终回答使用测试提供的阶段结果、保持同一身份。
- 底层模型身份问题回答“没有相关数据，无法提供确切答案”，没有编造供应商。应用当前未向模型提供可对外披露的供应商身份数据，因此本次不要求它报出实际模型名。
- 六个场景总耗时分别约 42.6、7.7、19.9、22.7、12.1、15.1 秒，包含本机准备和供应商调用，不能当作纯模型首 token 延迟。
- 图片路径仅做假模型验证，未调用真实视觉供应商。工作流真实冒烟验证最终回答入口，不运行整个研究流程。
- 原先记录的 2 项 Supervisor、12 项研究阶段失败不在本次回归集合内，本次未修改或重新归类这些基线失败。

复现：

```powershell
.\.venv\Scripts\python.exe -m pytest src/web_app/tests/test_gap_identity.py src/web_app/tests/test_chat_fast_path.py src/web_app/tests/test_conversation_document_chat.py src/web_app/tests/test_chat_control.py src/web_app/tests/test_chat_runtime_compatibility.py src/web_app/tests/test_llm_registry.py src/web_app/tests/test_summary_tasks.py -q

# 可选：读取已有模型配置，使用临时会话发起真实供应商请求
$env:PYTHONIOENCODING='utf-8'
$env:GAP_IDENTITY_SMOKE='1'
.\.venv\Scripts\python.exe -m pytest src/web_app/tests/test_gap_identity.py -q -s -k real_identity
```

重启后端后生效，开发环境启用自动重载时可直接加载。无需迁移、重新配置模型或修改前端。未提交、推送或部署。
