# 普通聊天停止与插话接续

本次实现只覆盖普通文本聊天的意图识别、只读上下文准备和生成阶段。采用同一 conversation 下中断旧 run、创建关联新 run 的方式，不恢复模型内部推理状态。未改研究、工具执行、审批恢复的内部流程，未增加运行中附件追加。

## 接口与状态

- `POST /agent/runs/{run_id}/interrupt`：`{"client_command_id":"UUID"}`。
- `POST /agent/runs/{run_id}/steer`：`{"client_command_id":"UUID","text":"只讲聊天模块，简单一点"}`。
- 两个接口校验登录用户归属；成功返回 HTTP 202，数据包含 `client_command_id`、`run_id`、`successor_run_id`、`kind`、`status`、`error`。
- `status` 为 `accepted`、`applied` 或 `failed`。202/accepted 仅代表持久化接收；中断事件与 applied 才代表控制生效。GET run 的 `controls` 可查询最终结果。
- 同一用户下按 `client_command_id` 去重；请求内容冲突返回 409。同一个请求重试可读取更新后的处理状态，不重复保存消息或执行生成。
- run 返回 `can_interrupt`、`can_steer`、`supersedes_run_id`。进入研究、工具、审批及其他非聊天阶段前关闭控制能力；状态切换与控制接收共享进程内锁。能力关闭后返回 409，前端保留输入。

插话在同一个事务中创建控制记录、新用户消息、queued 助手消息及接续 run。旧任务取消并关闭模型流、从事件账本保存部分回复后，才启动新任务。取消等待超过 2 秒则命令失败，接续 run 不启动，新消息保留。已经完成的 run 收到插话时按下一轮处理；停止操作不会创建新回复。

新 run 重新进行意图识别，沿用模型配置、现有历史及 Memory/GSSC。原问题、历次补充要求、最近一次未完成回复、最新消息作为接续上下文传入意图识别和最终生成。最近的部分回复最多取 4000 字符；明确修改优先，换话题则回答新话题。中断分支不调用完成后的摘要更新；未来历史和摘要输入明确标记未完成内容。

## 流式与恢复

前端保留 SSE，按 run/message 隔离输出，按事件游标去重。关闭页面或断开流只停止订阅，不取消后端执行。刷新重新读取消息、事件与控制结果，恢复部分回复、中断标记和接续关系。未知网络结果通过同一个命令 ID 重试确认。

后端取消令牌沿异步调用及只读线程传播，阻止旧任务继续写入输出事件或完成结果。只读线程中的阻塞调用不能被 Python 强制终止；它可以结束读取，但其后续事件写入被令牌拦截。供应商是否立即停止计算或计费不属于本地取消保证。

首次启动会将遗留的 enabled/stopping/queued 聊天执行标记为中断，保存账本内部分输出；未完成控制标记为失败，不自动重放。研究和审批 run 不进入本次聊天恢复分支。

## 数据库与运行前提

仅支持一个应用执行进程；不要使用多个 Uvicorn worker 执行这些接口。本次没有增加 Redis 队列或分布式调度。

增量迁移：`alembic/versions/20260907_0014_chat_control.py`，前置版本 `20260818_0013`。增加控制记录表，以及 run 的控制阶段、接续关联字段，不删除或重写旧会话。

本次未对现有数据库执行迁移。实际启用前，需要维护者停应用、核对当前迁移版本及前置迁移要求，再手动执行 `alembic upgrade 20260907_0014`。前置 LLM registry 迁移有其自身的活跃任务检查，不能跳过。启动新代码前必须完成所需 schema 更新。

本地只读检查发现 `llm_models`、`llm_connections` 表和 `user_profiles.default_llm_model_id` 缺失，因此未执行真实模型端到端冒烟测试，也没有通过自动迁移绕过此前提。数据库准备好、配置可用模型后，应使用新建隔离会话执行以下人工验收，不接入真实外部写工具：

1. 要求解释架构，首 token 前点击停止；确认没有新回复，刷新仍为中断。
2. 输出中发送“只讲聊天模块，简单一点”；确认部分旧回复保留，新回复遵循修改。
3. 新回复中继续发送补充要求，再发送换话题要求；确认消息顺序、run 接续关系和回答内容。
4. 输出中断开网络后重连、刷新；确认内容不重复，真实后端状态恢复。

## 自动验证记录（2026-09-07）

- 新增聊天控制后端测试：16 项通过。使用临时 SQLite 与假模型，覆盖首 token 前/中途取消、流关闭、意图调用取消、部分持久化、接续提示词、连续插话、完成后插话、幂等、越权、阶段拒绝、迟到写入、2 秒超时分支、重启恢复及迁移离线 SQL。
- 综合后端回归：116 项通过、6 项已有失败。
- 研究与审批专项抽样：14 项通过、1 项已有失败、1 项跳过。
- 前端事件投影：4 项通过，覆盖 run/message 隔离、游标重放去重、中断后迟到 token/完成事件、部分回复保留。
- `npm run build` 和后续 `npm run type-check` 通过；构建仍提示单个 bundle 超过 500 kB，本次未做无关拆包。
- 未完成浏览器和真实供应商的端到端联合验收；自动测试不证明真实模型必然遵循每条补充要求。

已有失败保持未修改：

| 文件 | 测试 | 已有原因 |
| --- | --- | --- |
| `test_agent_runtime_llm_supervisor.py` | `test_config_model_overrides_env` | 断言引用已不存在的 settings.model |
| 同上 | `test_model_name_is_passed_to_factory` | monkeypatch 引用不存在的 get_chat_model_by_name |
| `test_agent_runtime_p7c_tool_node_results.py` | `test_interrupt_pause_sets_approval_payload_and_interrupts` | 测试工具 ID `tc1` 无法转换为整数 |
| 同上 | `test_interrupt_resume_approved_clears_pending_and_completes` | 同上，工具结果与旧断言不符 |
| `test_agent_runtime_p3a_parallel_read_stage.py` | `test_rag_agent_reuses_prepared_evidence_without_recalling_search` | 缺少旧测试预期的 `_parallel_read_evidence_used` |
| 同上 | `test_rag_prepare_timeout_then_formal_rag_agent_falls_back` | 返回空回答，与旧 fallback 断言不符 |
| `test_research_stage5.py` | `test_open_deep_research_adapter_fallback_returns_result` | OpenDeepResearchAdapter 已不存在 fallback 属性 |

两项 Supervisor 失败在修改前记录；其余五项使用 `git archive HEAD` 导出的独立临时源码重跑，确认同名失败及原因一致。没有为使测试变绿修改这些功能。

核心测试命令：

```powershell
.venv/Scripts/python.exe -m pytest src/web_app/tests/test_chat_control.py -q
cd frontend
node --test tests/chat-control.test.mjs
npm run build
```

代码、迁移与测试保留在工作区供审查；没有提交、推送、部署，也没有修改原有上传文件或简历。
