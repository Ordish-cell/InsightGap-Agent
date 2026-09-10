# 会话彻底删除

本次实现使用持久化删除任务，限定单应用执行进程。没有自动执行迁移，也没有删除现有用户会话。归档仍可恢复；彻底删除不可恢复。

## 启用

停止后端，在项目根目录手动执行增量迁移，再按原命令启动后端。前端重新启动或重新构建。

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade 20260909_0015
```

此迁移接在 `20260907_0014` 后，只创建 `conversation_deletion_tasks`。缺失时彻底删除返回 503 和 `DELETION_MIGRATION_REQUIRED`，不会回退旧删除。测试中的迁移只运行在临时 PostgreSQL schema / SQLite 数据库。

## 接口与行为

- `DELETE /api/v1/agent/conversations/{id}/hard` 返回 202 和任务。重复调用返回同一任务。
- 待审批任务先返回 409；明确选择取消审批后用 `?cancel_pending=true`，不执行工具。
- `GET /api/v1/agent/deletion-tasks` 和 `GET /api/v1/agent/deletion-tasks/{id}` 查询当前用户任务。
- `POST /api/v1/agent/deletion-tasks/{id}/retry` 手动重试失败任务，重新获得一次执行及最多三次自动重试额度。以前的尝试数保留在内部进度记录。
- 旧软删除接口标记弃用，侧栏删除按钮统一使用彻底删除。

接受后将会话标记为 deleting，阻止新 run、插话、审批恢复及文件关联。普通聊天取消并等待最多两秒收尾，摘要任务等待最多三十秒；未退出则失败，保留数据和删除屏障。正在执行的研究／工具直接返回 409。

资源清单提交后，精确清理 Qdrant、checkpoint、应用文件；最后用单个数据库事务删除关联行并保存完成结果。外部错误、文件占用或数据库回滚均保留清单供重试。重启只恢复已接受删除任务，不重放聊天或工具。自动重试最多三次，包含准备阶段异常及重启恢复；用尽后需手动重试。

## 范围

删除当前会话摘要、摘要分段、消息、run 子记录、控制记录、run、明确属于该会话的工作记忆，以及独占 chat_upload 文档、切片、成果和研究结果。

保留 semantic、episodic 和显式 `visible_in_long_term_memory=true` 的记忆及向量。工作记忆按 conversation_id、run_id 或结构化 memory_id 归属；不根据时间或内容猜测。无法归属者保留并返回警告。

共享检查包含其他会话消息与 run、Skill 结构化引用、研究结果关联及重复文件路径。独立知识库文档、已独立保存或有明确共享引用的成果保留。保留成果和研究记录解除 SQL run 关联。只删除数据库登记且规范化后位于应用存储目录内的文件；不删除用户原文件，不递归清空目录。

Qdrant 覆盖 dense、hybrid、临时记忆、摘要分段 collection，限定用户及资源 ID，等待完成并复查剩余数量。checkpoint 按清单中的运行 thread ID 删除。若启用了 Neo4j 记忆投影，还精确清理选中临时记忆的 UserMemory 投影；共享概念节点保留，服务错误会使删除失败。

成功后清空详细资源清单和路径，只留任务凭据、计数及保留原因。前端轮询真实任务状态，显示停止中、删除中、失败重试或完成；刷新后恢复任务，完成提示可关闭。

## 验证记录（2026-09-09）

环境：Windows、本项目虚拟环境、真实 PostgreSQL 的随机隔离 schema、开启外键的 SQLite、真实 Qdrant 的四个随机隔离 collection 及 Qdrant 本地引擎。测试资源完成后按准确名称清理，未清理用户数据。

- 相关集成与回归：117 passed、2 skipped；包括删除、聊天控制、后台摘要、文件跨轮问答、审批、记忆。随后独占成果删除与独立 Skill 保留补测 2 passed，合计 119 项通过。跳过项分别为 SQLite 不适用的 PostgreSQL checkpoint 测试及既有可选测试。
- Qdrant 在线验证已通过：目标删除、其他用户与非目标数据保留、重复清理幂等。
- PostgreSQL 验证了摘要消息外键顺序、事务回滚、增量迁移、checkpoint 按 thread 精确删除及重复执行。
- API 验证 202、任务去重、用户隔离、重新查询状态和手动重试；还验证准备阶段重试上限、停止与摘要等待、删除屏障和陈旧对象写回拦截。
- 前端 TypeScript 检查通过。
- 原有失败：Supervisor 2 项（旧模型设置接口断言），research_stage5 12 项（旧 adapter 契约和测试模型配置）。本次未修改这些测试或为它们绕开模型配置。

复现相关测试：

```powershell
$env:DELETION_POSTGRES_TEST='1'
$env:DELETION_QDRANT_TEST='1'
.\.venv\Scripts\python.exe -m pytest src/web_app/tests/test_conversation_deletion.py src/web_app/tests/test_chat_control.py src/web_app/tests/test_summary_tasks.py src/web_app/tests/test_conversation_document_chat.py src/web_app/tests/test_approval_expiry.py src/web_app/tests/test_memory_service.py src/web_app/tests/test_conversation_memory.py -q
```

## 验证限制

未用真实用户会话做浏览器破坏性验收；前端目前验证到类型检查和对应 API 集成。Neo4j 在线清理、操作系统文件占用及实际强制杀进程恢复尚未做在线故障注入。分布式执行不受支持；跨 PostgreSQL、Qdrant、文件系统不提供原子事务保证，失败后的会话保持不可用并通过任务重试继续清理。归属不明或不能确认独占的历史资产有意保留并报告，不承诺将它们清空。
