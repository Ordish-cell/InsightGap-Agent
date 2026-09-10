# 普通聊天短路径与实时执行反馈

## 使用与回退

重启后端并刷新首页聊天。`CHAT_FAST_PATH_ENABLED` 默认 `true`；在 `.env` 设置为 `false` 并重启，可回退到原意图识别及编排路径，仍使用新的真实事件展示。本次无需新增数据库迁移，不要为了启用本功能重建数据库。

首页 `/` 是日常聊天入口；`/agent` 是原有 Runtime 调试页，会显示调试信息。

普通聊天在权限检查后，仅准备近期消息和已有摘要，复用 GSSC 预算，然后用当前会话模型的一次流式调用完成分流与回答。需要检索、研究、工具、上传内容或明确操作的请求进入原有工作流。运行、消息、停止与接续机制共用，没有新增执行队列或独立 Worker。

## 路由协议

系统消息要求首行输出 `chat` 或 `workflow`。后端最多缓冲首行 512 字符；首行不写入回答、消息或前端。`chat` 后的正文按模型真实 chunk 推送；`workflow` 关闭入口流后继续原有意图识别，同一 run 不重复创建消息。

`workflow` 单独出现在正常流末尾，即使没有末尾换行，也视为完整路由行。只有 `chat` 而没有正文、非法首行、超长首行等情况，记录 `chat_route_fallback` 并回退一次。供应商异常不是协议回退，不额外重试整个聊天路径。工具权限仍由原有安全机制决定。

SDK 首次导入及客户端构造放在线程执行，避免冷启动阻塞停止操作与 SSE；冷启动耗时本身仍然存在。没有更换用户模型，也没有自动改推理强度。

## 事件与展示

- 新消息及创建事件带 `interaction_version: 2`；旧账本不重写。
- `interaction_mode` 表示 `pending / chat / workflow`；兼容回退在原 Planner 确认意图后更新展示模式。
- `node_started / node_completed / node_failed / node_cancelled / node_paused` 包含独立 `step_id`、`parent_step_id`、展示名称、状态和可用耗时。开始事件在执行前产生；节点重试有新的步骤标识。
- `progress_delta / progress_completed` 使用 `run_id + block_id` 合并，关联 `step_id`；完成事件的正文替换该段增量内容，可附带来源引用。目前工作流接入点发布节点已经产出的公开结果或检索结果事实，不额外调用模型润色过程。
- 普通聊天不展示工作时间线。复杂任务保留一个当前操作、实际发现段落和默认折叠的执行详情。旧模板文本不会混入新发现。
- 事件先持久化，再通知进程内 SSE 订阅者；保留 250 毫秒轮询兜底。刷新按账本游标恢复；终态后的迟到进展不再更新界面。
- 正常完成事件已经包含完整响应时，前端直接使用该响应，不再等待一次额外 GET 才显示完成。

本次不展示模型内部推理，不改研究内部调度。只有节点边界暴露结果的任务，发现会在该结果实际产生后展示；仍含同步阻塞调用的旧工具/研究内部步骤，不保证任意时刻都有新反馈。

## 后台记忆

回答落库、完成事件发送后才调度摘要。单进程最多并发两个记忆任务，同一会话串行，重复调度合并。线程使用独立数据库会话和对应模型上下文，不继承旧 run 的取消令牌。

摘要依据 `last_message_id` 补齐尚未覆盖的已完成轮次，复用 `summary_version`。中断或失败助手回复不进入摘要及历史分段。下一次聊天直接使用最近消息和已经提交的摘要，不等待后台更新。摘要失败保留旧版本，后续调度再补齐。应用关闭等待正在运行的维护任务，单次模型请求受原供应商超时及重试配置约束。

仍限定单应用执行进程；没有分布式一致性保证。进程重启不会重放聊天任务，未覆盖的消息在后续摘要调度时处理。

## 验证记录（2026-09-09）

环境：Windows、本项目 Python 虚拟环境、隔离 SQLite 数据库、Node 20、Vite、无头 Microsoft Edge。以下数据是本地测试样本，不是生产性能承诺。

| 验证 | 结果 |
| --- | --- |
| 假模型完整聊天，30 次，模型调用前准备 p95 | 185.4 ms |
| 同批次回答保存后至完成事件 p95 | 11.4 ms |
| 模拟账本 + 真实 HTTP SSE + 完整首页，30 次推送到绘制 p95 | 13 ms |
| 独立进展组件，30 次事件注入到绘制 p95 | 9 ms |
| 前端状态投影测试 | 11 项通过，含审批恢复与中断后输出隔离 |
| 前端类型检查与生产构建 | 通过；保留原有大于 500 kB 的包体积提示 |

SQLite 默认时间戳只有秒级精度，性能测试使用提交点的单调时钟，避免把跨秒误判为 1000 ms。

真实供应商冒烟读取当前已配置模型 `qwen3.7-flash`，只向隔离 SQLite 写入测试会话：

- “唉，我好累”：选择 chat，同一条模型流生成 137 字回答，总计约 21.6 秒；其中 SDK 冷初始化约 9.2 秒，从开始请求模型到首正文约 11.5 秒。
- “请联网核查今天的科技新闻，引用来源”：约 5.0 秒选择 workflow；测试在转交边界停止，没有访问真实搜索或外部写工具。

因此本次已经消除普通聊天强制串行分类、预取和摘要尾部等待，但没有把外部模型或 SDK 冷启动变成瞬时完成。

组合回归结果：**149 通过、17 个既有失败、1 个默认跳过的真实供应商测试**；真实供应商测试单独显式启用后通过。17 个失败已在独立 `git archive HEAD` 副本中复现：

- Supervisor 2 项：旧测试依赖已移除的 `model` 字段和按名称取模型接口。
- 旧工具事件测试 1 项：调用签名缺少数据库参数。
- Research stage5 12 项：旧 fallback 属性、10 项缺失模型配置的夹具，以及旧健康状态预期。
- RAG 并行准备 2 项：旧 prepared-evidence 标记及 fallback 答案预期。

未通过删除断言或修改研究/审批内部逻辑消除这些失败。测试中发现的历史分段过滤兼容问题已修复，55 项会话摘要/分段测试通过。

## 复验命令

```powershell
.\.venv\Scripts\python.exe -m pytest src/web_app/tests/test_chat_fast_path.py src/web_app/tests/test_summary_tasks.py src/web_app/tests/test_chat_live_integration.py src/web_app/tests/test_chat_control.py src/web_app/tests/test_chat_runtime_compatibility.py src/web_app/tests/test_agent_event_ledger.py -q
npm --prefix frontend run build
$testFiles = @(Get-ChildItem frontend/tests -Filter '*.test.mjs' | ForEach-Object { $_.FullName })
node --test $testFiles
```

浏览器脚本 `frontend/tests/chat-live.browser.mjs` 使用本地 Vite 和完全隔离的模拟 API/SSE 服务，`live-progress.browser.mjs` 检查组件及窄屏。需要环境可解析 Playwright；设置 `CHAT_UI_URL` 指向本地 Vite 地址后运行。真实模型测试默认跳过，只有显式设置 `CHAT_REAL_SMOKE=1` 才运行。

本次未提交、推送或部署，未对现有数据库执行迁移；工作区已有改动保留。
