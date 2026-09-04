"""Build the offline RAG evaluation v2 corpus and gold cases.

The generated dataset is deterministic and does not require PostgreSQL,
Qdrant, an embedding model, or an LLM.  It intentionally contains similar
versions, conflicting values, and unrelated distractors so top-k retrieval is
not rewarded merely for finding a shared keyword.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "src" / "web_app" / "tests" / "fixtures" / "rag_eval_v2"
CORPUS_DIR = DATASET_DIR / "corpus"

CATEGORY_TEST_COUNTS = {
    "exact": 6,
    "table": 5,
    "semantic": 6,
    "parent_context": 5,
    "conflict": 3,
    "summary": 3,
    "unanswerable": 2,
}

EXPECTED_CATEGORY_COUNTS = {
    "exact": 20,
    "table": 15,
    "semantic": 20,
    "parent_context": 15,
    "conflict": 10,
    "summary": 10,
    "unanswerable": 10,
}


DOCUMENTS: dict[str, str] = {
    "contract_2024_archive.md": """# 采购服务合同（2024 归档版）

状态：已归档，不再适用于新订单。

## 合同信息

- 合同编号：HT-2024-017
- 签署日期：2024-03-02
- 合同总额：人民币 96,000 元
- 联系邮箱：contract-2024@example.com

## 付款与退款

首付款比例为 40%，验收后支付剩余 60%。未交付部分可在付款后 14 天内申请退款。

## 争议处理

争议由北京仲裁委员会处理。
""",
    "contract_2025_revision.md": """# 采购服务合同（2025 修订版）

状态：历史版本，已被 2026 最终版取代。

## 合同信息

- 合同编号：HT-2025-041
- 签署日期：2025-04-18
- 合同总额：人民币 112,500 元
- 联系邮箱：contract-2025@example.com

## 付款与退款

首付款比例为 30%，交付后支付 70%。未交付部分的退款期限为付款后 21 天。

## 争议处理

争议由杭州仲裁委员会处理。
""",
    "contract_2026_draft.md": """# 采购服务合同（2026 草案）

状态：草案，仅供讨论，不具有最终效力。

## 合同信息

- 草案编号：DRAFT-HT-2026-083
- 拟签署日期：2026-05-20
- 草案总额：人民币 120,000 元
- 草案联系邮箱：draft-contract@example.com

## 付款与退款草案

草案建议首付款比例为 35%，退款期限为付款后 20 天。

## 争议处理草案

草案建议由深圳国际仲裁院处理。
""",
    "contract_2026_final.md": """# 采购服务合同（2026 最终生效版）

状态：最终版；生效日期为 2026-06-01。该版本取代 2026 草案及所有历史版本。

## 合同信息

- 合同编号：HT-2026-083
- 签署日期：2026-06-01
- 合同总额：人民币 128,000 元
- 联系邮箱：contract-final@example.com

## 付款与退款

首付款比例为 25%，验收通过后支付剩余 75%。未交付部分可在付款后 30 天内申请退款，已交付部分不退款。

## 争议处理

争议由上海仲裁委员会处理。提交仲裁前，双方应先进行 10 个工作日的书面协商。
""",
    "rag_architecture_v1.md": """# RAG Architecture V1（历史方案）

状态：deprecated。

## 检索

V1 仅使用 Python BM25，配置键为 `RAG_BACKEND=python_bm25`，索引名称为 `rag_chunks_v1`。

## 切分

所有文档按 500 tokens 固定长度切分，不保存 parent 层级，overlap 为 50 tokens。

## 已知问题

同义改写召回较弱，长段落被切断后缺少回答上下文。
""",
    "rag_architecture_v2.md": """# RAG Architecture V2（当前生产方案）

状态：active；版本 2.3。

## 检索链路

当前方案组合 dense embedding 与 sparse lexical retrieval，在 Qdrant 中使用 RRF 融合排序。主配置键是 `RAG_HYBRID_BACKEND=qdrant_hybrid`，集合名称为 `rag_hybrid_v2`，RRF 参数 k=60。

## Parent-Child Chunking

Parent 目标长度为 800 tokens；Child 目标长度为 180 tokens，overlap 为 40 tokens。Child 用于精确召回，命中后通过 parent_id 回查 Parent 作为回答上下文。

## 降级策略

Qdrant 不可用时回退到 `python_bm25`。回退必须产生 `qdrant_hybrid_failed` warning，不允许静默伪装成 Hybrid 成功。

## 质量门禁

冻结测试集要求 Evidence Recall@5 不低于 0.90，MRR@10 不低于 0.75，并单独报告 P95 检索延迟。
""",
    "deployment_runbook.md": """# Agent 平台部署手册

## 健康检查

HTTP 健康检查地址为 `/api/v1/health`。服务正常时返回 `status=ok`。

## 发布顺序

先执行 `alembic upgrade head`，再启动 FastAPI API，最后启动 worker。前端构建命令是 `npm run build`。

## 回滚

应用回滚命令为 `deployctl rollback --service agent-api --to previous`。数据库迁移默认不自动降级，必须执行单独审核。

## 性能目标

预热后的检索 P95 目标为 2.5 秒，首个 SSE 事件目标为 800 毫秒以内。
""",
    "tool_security_policy.md": """# Agent 工具安全策略

## 风险等级

- L0：纯计算，不访问外部数据。
- L1：读取用户已授权的内部数据。
- L2：读取互联网或受限资源，需要审计但不需要人工批准。
- L3：发送邮件、写文件或提交表单等外部写操作，执行前必须人工审批。
- L4：删除数据、转账或不可逆操作，默认阻断。

## 幂等与审计

每次工具调用必须包含 idempotency_key，并记录 tool_name、输入摘要、风险等级、审批状态和执行结果。

## 审批恢复

L3 工具暂停后通过 checkpoint 保存状态；批准后使用相同 run thread_id 恢复，副作用只能执行一次。
""",
    "inventory_north.csv": """product_model,price,quantity,owner,email,status,warehouse
NX-100,8999,18,Alice,alice.north@example.com,in_stock,North
NX-200,12999,7,Bob,bob.north@example.com,low_stock,North
NX-300,15999,0,Carol,carol.north@example.com,out_of_stock,North
AX-9,6999,31,Dylan,dylan.north@example.com,in_stock,North
""",
    "inventory_south.csv": """product_model,price,quantity,owner,email,status,warehouse
NX-100,8799,9,Erin,erin.south@example.com,low_stock,South
NX-200,12599,22,Frank,frank.south@example.com,in_stock,South
NX-300,15499,5,Grace,grace.south@example.com,in_stock,South
AX-9,7199,0,Henry,henry.south@example.com,out_of_stock,South
""",
    "pricing_archive_2025.csv": """product_model,price,effective_date,status
NX-100,8299,2025-01-01,expired
NX-200,11999,2025-01-01,expired
NX-300,14999,2025-01-01,expired
AX-9,6599,2025-01-01,expired
""",
    "incidents_2026_q1.csv": """incident_id,service,severity,duration_minutes,status,owner
INC-2026-021,qdrant,SEV-2,47,resolved,Li Ming
INC-2026-034,postgres,SEV-1,83,resolved,Wang Lei
INC-2026-052,embedding-api,SEV-2,36,resolved,Zhao Xin
INC-2026-067,frontend-sse,SEV-3,19,resolved,Chen Yu
""",
    "incident_qdrant_2026_02.md": """# INC-2026-021 Qdrant 检索异常复盘

## 影响

2026-02-11 10:03 至 10:50，Hybrid 检索出现 502，持续 47 分钟。18% 的查询回退到 BM25，没有发生跨用户数据泄漏。

## 根因

集合 schema 升级后只完成了部分重新摄入，部分 point 缺少 sparse vector；网关健康检查仍然只检查 HTTP 进程存活。

## 修复

完成全量重新摄入，增加 dense/sparse vector 数量一致性检查，并将 collection readiness 加入发布门禁。

## 负责人

事故负责人为 Li Ming，复盘截止日期为 2026-02-14。
""",
    "incident_postgres_2026_03.md": """# INC-2026-034 PostgreSQL 连接池耗尽复盘

## 影响

2026-03-08 14:20 至 15:43，Agent run 创建接口大量超时，事故持续 83 分钟，严重级别为 SEV-1。

## 根因

SSE 客户端断开后，两个异常分支没有归还数据库 Session，连接池逐渐耗尽。

## 修复

统一使用 context manager 管理 Session，增加连接池使用率告警，并为 SSE 断连路径加入集成测试。

## 负责人

事故负责人为 Wang Lei。
""",
    "incident_embedding_2026_04.md": """# INC-2026-052 Embedding API 延迟复盘

## 影响

2026-04-03 09:12 至 09:48，Embedding API P95 从 680 毫秒升至 6.2 秒，影响持续 36 分钟，严重级别为 SEV-2。

## 根因

批处理上限被错误配置为 1，导致重新摄入任务产生大量串行请求；重试没有 jitter，进一步放大流量峰值。

## 修复

批处理上限恢复为 32，指数退避加入随机 jitter，后台摄入与在线查询使用独立并发额度。

## 负责人

事故负责人为 Zhao Xin。
""",
    "refund_policy_2024.md": """# 客户退款政策（2024 旧版）

状态：expired。

## 时限

数字服务未交付部分可在付款后 14 天内退款。硬件产品签收后 7 天内可申请退货。

## 手续费

非质量问题退款收取 8% 服务费。

## 联系方式

旧版退款邮箱为 refund-legacy@example.com。
""",
    "refund_policy_2026.md": """# 客户退款政策（2026 当前版）

状态：active；生效日期 2026-01-15，本文件取代 2024 旧版。

## 时限

数字服务未交付部分可在付款后 30 天内退款。硬件产品签收后 15 天内可申请退货。

## 手续费

未交付数字服务不收手续费；非质量问题的硬件退货收取 5% 整备费。

## 联系方式

当前退款邮箱为 refund-2026@example.com，处理时限为 3 个工作日。
""",
    "employee_travel_policy.md": """# 员工差旅政策

## 交通

单程高铁不超过 5 小时时应优先选择高铁。经济舱机票需提前至少 7 天预订。

## 住宿

一线城市住宿上限为每晚 650 元，其他城市为每晚 450 元。超出标准需要直属经理书面批准。

## 报销

差旅结束后 10 个工作日内提交发票和行程单，财务邮箱为 travel-finance@example.com。
""",
    "product_launch_plan.md": """# Orion 产品发布计划

## 时间线

Orion 内测日期为 2026-09-10，公开发布日期为 2026-10-08。冻结代码日期为 2026-09-25。

## 成功指标

发布后 30 天内目标是 2,000 个激活团队，次日留存率达到 45%，核心工作流成功率达到 97%。

## 风险

主要风险是移动端审批体验尚未完成、Embedding 成本超预算，以及英文文档召回率低于中文文档。

## 负责人

产品负责人为 Sun Yi，技术负责人为 Gao Fei。
""",
    "memory_design_v1.md": """# Memory 设计 V1（已停用）

状态：deprecated。

## 方案

V1 将最近 50 轮原始对话全部写入 prompt，并把每条消息都作为长期记忆保存。

## 问题

该方案造成 token 浪费、闲聊污染和跨任务错误召回，无法区分运行状态与用户长期偏好。
""",
    "memory_design_v2.md": """# Memory 设计 V2（当前方案）

状态：active。

## 三层记忆

Working Memory 保存当前任务临时状态；Episodic Memory 保存具有时间和行为背景的经历；Semantic Memory 保存稳定偏好与事实。

## 上下文选择

GSSC 按 Gather、Select、Structure、Compress 四阶段组装上下文。候选内容按相关性、重要性和 token budget 选择，不把所有记忆直接塞入 prompt。

## 对话压缩

每 20 条消息冻结一个历史 segment，同时维护 running summary。原始消息仍留在会话存储中，但不会全部进入模型上下文。
""",
    "chinese_risk_notes_v2.md": """# 混合检索上线风险说明

## 背景

本说明用于评估 Qdrant Hybrid Search 从灰度切换为默认检索后端的风险。

## 风险

第一，部分重新摄入会造成 sparse vector 缺失，使结果静默偏向 dense。第二，中文短查询可能产生过多二元词，导致无关文档排名升高。第三，Parent 回查失败会让正确 Child 缺少回答上下文。

## 缓解措施

上线前核对 dense/sparse point 数量，按 exact、table、semantic、summary 分桶报告指标，并在 Parent 回查失败时记录 `parent_context_missing` warning。

## 决策

只有冻结测试集 MRR@10 达到 0.75 且无答案误召回率不高于 10% 时，才允许全量切换。
""",
}


def evidence(filename: str, heading: str, *facts: str) -> list[dict[str, Any]]:
    return [{"filename": filename, "heading": heading, "required_facts": list(facts), "relevance": 3}]


def case(
    category: str,
    question: str,
    answer: str | None,
    docs: list[str],
    gold_evidence: list[dict[str, Any]],
    *,
    aliases: list[str] | None = None,
    negatives: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "question": question,
        "answerable": answer is not None,
        "expected_behavior": "answer" if answer is not None else "abstain",
        "gold_answer": answer,
        "answer_aliases": aliases or ([] if answer is None else [answer]),
        "relevant_documents": docs,
        "gold_evidence": gold_evidence,
        "hard_negative_documents": negatives or [],
        "tags": tags or [],
    }


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    exact_specs = [
        ("2026 最终生效合同的合同编号是什么？", "HT-2026-083", "contract_2026_final.md", "合同信息", ["contract_2026_draft.md", "contract_2025_revision.md"]),
        ("2026 最终版合同总额是多少？", "人民币 128,000 元", "contract_2026_final.md", "合同信息", ["contract_2026_draft.md"]),
        ("2026 最终版合同的签署日期是哪一天？", "2026-06-01", "contract_2026_final.md", "合同信息", ["contract_2026_draft.md"]),
        ("最终合同的联系邮箱是什么？", "contract-final@example.com", "contract_2026_final.md", "合同信息", ["contract_2025_revision.md"]),
        ("2025 修订版合同编号是什么？", "HT-2025-041", "contract_2025_revision.md", "合同信息", ["contract_2026_final.md"]),
        ("2024 归档合同的总额是多少？", "人民币 96,000 元", "contract_2024_archive.md", "合同信息", ["contract_2026_final.md"]),
        ("RAG V2 使用的 Qdrant collection 名称是什么？", "rag_hybrid_v2", "rag_architecture_v2.md", "检索链路", ["rag_architecture_v1.md"]),
        ("RAG V2 的 RRF 参数 k 是多少？", "60", "rag_architecture_v2.md", "检索链路", ["rag_architecture_v1.md"]),
        ("平台健康检查 endpoint 是什么？", "/api/v1/health", "deployment_runbook.md", "健康检查", []),
        ("应用回滚使用什么命令？", "deployctl rollback --service agent-api --to previous", "deployment_runbook.md", "回滚", []),
        ("Qdrant 事故编号是什么？", "INC-2026-021", "incident_qdrant_2026_02.md", "标题", ["incidents_2026_q1.csv"]),
        ("PostgreSQL 连接池事故持续了多少分钟？", "83 分钟", "incident_postgres_2026_03.md", "影响", ["incidents_2026_q1.csv"]),
        ("Embedding API 事故负责人是谁？", "Zhao Xin", "incident_embedding_2026_04.md", "负责人", ["incidents_2026_q1.csv"]),
        ("2026 当前退款政策的联系邮箱是什么？", "refund-2026@example.com", "refund_policy_2026.md", "联系方式", ["refund_policy_2024.md"]),
        ("一线城市住宿每晚上限是多少？", "650 元", "employee_travel_policy.md", "住宿", []),
        ("Orion 公开发布日期是什么时候？", "2026-10-08", "product_launch_plan.md", "时间线", []),
        ("Orion 的技术负责人是谁？", "Gao Fei", "product_launch_plan.md", "负责人", []),
        ("Memory V2 每多少条消息冻结一个历史 segment？", "20 条消息", "memory_design_v2.md", "对话压缩", ["memory_design_v1.md"]),
        ("RAG V2 的 Child 目标长度是多少？", "180 tokens", "rag_architecture_v2.md", "Parent-Child Chunking", ["rag_architecture_v1.md"]),
        ("L4 工具默认如何处理？", "默认阻断", "tool_security_policy.md", "风险等级", []),
    ]
    for question, answer, filename, heading, negatives in exact_specs:
        cases.append(case("exact", question, answer, [filename], evidence(filename, heading, answer), negatives=negatives, tags=["deterministic"]))

    table_specs = [
        ("北仓 NX-100 的价格是多少？", "8999", "inventory_north.csv", "row:NX-100", ["inventory_south.csv", "pricing_archive_2025.csv"]),
        ("南仓 NX-100 还有多少库存？", "9", "inventory_south.csv", "row:NX-100", ["inventory_north.csv"]),
        ("北仓 NX-200 的库存负责人是谁？", "Bob", "inventory_north.csv", "row:NX-200", ["inventory_south.csv"]),
        ("南仓 NX-200 当前状态是什么？", "in_stock", "inventory_south.csv", "row:NX-200", ["inventory_north.csv"]),
        ("北仓 NX-300 的库存数量是多少？", "0", "inventory_north.csv", "row:NX-300", ["inventory_south.csv"]),
        ("南仓 NX-300 的售价是多少？", "15499", "inventory_south.csv", "row:NX-300", ["pricing_archive_2025.csv"]),
        ("北仓 AX-9 的负责人邮箱是什么？", "dylan.north@example.com", "inventory_north.csv", "row:AX-9", ["inventory_south.csv"]),
        ("南仓 AX-9 当前是否缺货？", "out_of_stock", "inventory_south.csv", "row:AX-9", ["inventory_north.csv"]),
        ("2025 归档价格表中 NX-200 的价格是多少？", "11999", "pricing_archive_2025.csv", "row:NX-200", ["inventory_north.csv", "inventory_south.csv"]),
        ("事故表里严重级别为 SEV-1 的服务是什么？", "postgres", "incidents_2026_q1.csv", "row:INC-2026-034", []),
        ("INC-2026-052 持续了多少分钟？", "36", "incidents_2026_q1.csv", "row:INC-2026-052", ["incident_embedding_2026_04.md"]),
        ("frontend-sse 事故的负责人是谁？", "Chen Yu", "incidents_2026_q1.csv", "row:INC-2026-067", []),
        ("南仓库存最多的产品型号是什么？", "NX-200", "inventory_south.csv", "rows", ["inventory_north.csv"]),
        ("北仓价格最高的产品型号是什么？", "NX-300", "inventory_north.csv", "rows", ["pricing_archive_2025.csv"]),
        ("北仓和南仓 NX-100 的价差是多少？", "200", "inventory_north.csv", "row:NX-100", ["pricing_archive_2025.csv"]),
    ]
    for question, answer, filename, heading, negatives in table_specs:
        gold_docs = [filename]
        gold = evidence(filename, heading, answer)
        if "北仓和南仓" in question:
            gold_docs = ["inventory_north.csv", "inventory_south.csv"]
            gold = evidence("inventory_north.csv", "row:NX-100", "8999") + evidence("inventory_south.csv", "row:NX-100", "8799")
        cases.append(case("table", question, answer, gold_docs, gold, negatives=negatives, tags=["structured_data"]))

    semantic_specs = [
        ("当前 RAG 为什么同时保留语义检索和关键词检索？", "同时覆盖语义改写与精确词项，并通过 RRF 融合排序。", "rag_architecture_v2.md", "检索链路", ["dense embedding", "sparse lexical retrieval", "RRF"], ["rag_architecture_v1.md"]),
        ("为什么不能把 Qdrant 故障后的结果继续标成 Hybrid？", "因为回退必须显式记录 warning，不能静默伪装成 Hybrid 成功。", "rag_architecture_v2.md", "降级策略", ["qdrant_hybrid_failed", "不允许静默"], []),
        ("Parent-Child 设计分别解决了什么问题？", "Child 负责精确召回，Parent 负责补充回答上下文。", "rag_architecture_v2.md", "Parent-Child Chunking", ["Child 用于精确召回", "Parent 作为回答上下文"], ["rag_architecture_v1.md"]),
        ("旧版 RAG 对同义表达为什么表现不好？", "因为 V1 只使用 BM25，缺少语义检索。", "rag_architecture_v1.md", "已知问题", ["同义改写召回较弱", "仅使用 Python BM25"], ["rag_architecture_v2.md"]),
        ("为什么数据库迁移不能跟着应用一起自动回滚？", "数据库迁移默认不自动降级，必须单独审核。", "deployment_runbook.md", "回滚", ["数据库迁移默认不自动降级", "单独审核"], []),
        ("哪些外部操作需要人在执行前确认？", "发送邮件、写文件、提交表单等 L3 外部写操作。", "tool_security_policy.md", "风险等级", ["L3", "人工审批"], []),
        ("工具调用为什么需要幂等键？", "用于避免审批恢复或重试时重复执行副作用。", "tool_security_policy.md", "幂等与审计", ["idempotency_key", "副作用只能执行一次"], []),
        ("Qdrant 502 的根本原因是什么？", "集合升级后只完成部分重新摄入，导致部分 point 缺少 sparse vector。", "incident_qdrant_2026_02.md", "根因", ["部分重新摄入", "缺少 sparse vector"], []),
        ("为什么 PostgreSQL 连接池最终会耗尽？", "SSE 断开后的异常分支没有归还数据库 Session。", "incident_postgres_2026_03.md", "根因", ["SSE 客户端断开", "没有归还数据库 Session"], []),
        ("Embedding 延迟事故为何被重试进一步放大？", "重试没有 jitter，导致请求在相近时间再次涌入。", "incident_embedding_2026_04.md", "根因", ["重试没有 jitter", "放大流量峰值"], []),
        ("当前数字服务退款是否收手续费？", "未交付数字服务退款不收手续费。", "refund_policy_2026.md", "手续费", ["不收手续费"], ["refund_policy_2024.md"]),
        ("坐高铁还是飞机由什么规则决定？", "单程高铁不超过 5 小时时优先选择高铁。", "employee_travel_policy.md", "交通", ["不超过 5 小时", "优先选择高铁"], []),
        ("Orion 发布面临哪些主要风险？", "移动端审批未完成、Embedding 成本超预算、英文文档召回率偏低。", "product_launch_plan.md", "风险", ["移动端审批", "Embedding 成本", "英文文档召回率"], []),
        ("旧版 Memory 为什么容易污染上下文？", "因为它把每条消息都保存为长期记忆，并将最近 50 轮全部放入 prompt。", "memory_design_v1.md", "问题", ["每条消息", "最近 50 轮", "闲聊污染"], ["memory_design_v2.md"]),
        ("GSSC 如何控制进入模型的记忆数量？", "按相关性、重要性和 token budget 选择候选内容。", "memory_design_v2.md", "上下文选择", ["相关性", "重要性", "token budget"], []),
        ("Working、Episodic、Semantic 三类记忆如何分工？", "分别保存当前任务状态、带时间行为背景的经历、稳定偏好与事实。", "memory_design_v2.md", "三层记忆", ["当前任务临时状态", "时间和行为背景", "稳定偏好与事实"], []),
        ("混合检索上线前为什么要核对两类向量数量？", "为了发现部分重新摄入造成的 sparse vector 缺失。", "chinese_risk_notes_v2.md", "缓解措施", ["dense/sparse point 数量", "sparse vector 缺失"], []),
        ("短中文查询可能给排序带来什么问题？", "可能产生过多二元词，使无关文档排名升高。", "chinese_risk_notes_v2.md", "风险", ["过多二元词", "无关文档排名升高"], []),
        ("Parent 回查失败会造成什么后果？", "正确 Child 会缺少用于回答的完整上下文。", "chinese_risk_notes_v2.md", "风险", ["Parent 回查失败", "缺少回答上下文"], []),
        ("发布后如何判断 Orion 核心流程是否达标？", "核心工作流成功率需要达到 97%。", "product_launch_plan.md", "成功指标", ["核心工作流成功率", "97%"], []),
    ]
    for question, answer, filename, heading, facts, negatives in semantic_specs:
        cases.append(case("semantic", question, answer, [filename], evidence(filename, heading, *facts), negatives=negatives, tags=["paraphrase"]))

    parent_specs = [
        ("2026 最终合同的付款结构和退款限制分别是什么？", "首付 25%，验收后付 75%；未交付部分 30 天内可退款，已交付部分不退款。", "contract_2026_final.md", "付款与退款", ["25%", "75%", "30 天", "已交付部分不退款"], ["contract_2026_draft.md"]),
        ("2026 最终合同仲裁前需要先做什么，持续多久？", "先进行 10 个工作日的书面协商。", "contract_2026_final.md", "争议处理", ["10 个工作日", "书面协商"], []),
        ("RAG V2 的 Parent、Child 和 overlap 参数分别是多少？", "Parent 800 tokens，Child 180 tokens，overlap 40 tokens。", "rag_architecture_v2.md", "Parent-Child Chunking", ["800 tokens", "180 tokens", "40 tokens"], ["rag_architecture_v1.md"]),
        ("RAG V2 故障时如何降级并留下什么信号？", "回退到 python_bm25，并产生 qdrant_hybrid_failed warning。", "rag_architecture_v2.md", "降级策略", ["python_bm25", "qdrant_hybrid_failed"], []),
        ("平台发布的数据库、API、worker 启动顺序是什么？", "先迁移数据库，再启动 FastAPI API，最后启动 worker。", "deployment_runbook.md", "发布顺序", ["alembic upgrade head", "FastAPI API", "worker"], []),
        ("L3 工具批准后如何保证不会重复产生副作用？", "使用相同 run thread_id 从 checkpoint 恢复，并通过 idempotency_key 保证副作用只执行一次。", "tool_security_policy.md", "审批恢复", ["checkpoint", "run thread_id", "idempotency_key", "执行一次"], []),
        ("Qdrant 事故的影响比例和持续时间是多少？", "18% 查询回退到 BM25，持续 47 分钟。", "incident_qdrant_2026_02.md", "影响", ["18%", "47 分钟"], []),
        ("Qdrant 事故修复包含哪两类检查？", "检查 dense/sparse vector 数量一致性，并检查 collection readiness。", "incident_qdrant_2026_02.md", "修复", ["dense/sparse vector 数量一致性", "collection readiness"], []),
        ("PostgreSQL 事故修复如何同时覆盖代码和监控？", "使用 context manager 管理 Session，并增加连接池使用率告警。", "incident_postgres_2026_03.md", "修复", ["context manager", "连接池使用率告警"], []),
        ("Embedding 事故后批处理和重试策略分别如何修改？", "批处理上限恢复到 32，指数退避增加随机 jitter。", "incident_embedding_2026_04.md", "修复", ["32", "指数退避", "jitter"], []),
        ("2026 退款政策对数字服务和硬件分别规定了多久？", "数字服务未交付部分 30 天，硬件签收后 15 天。", "refund_policy_2026.md", "时限", ["30 天", "15 天"], ["refund_policy_2024.md"]),
        ("差旅住宿超标时需要什么批准，之后多久提交报销？", "需要直属经理书面批准，并在差旅结束后 10 个工作日内报销。", "employee_travel_policy.md", "住宿与报销", ["直属经理书面批准", "10 个工作日"], []),
        ("Orion 从代码冻结到公开发布间隔多少天？", "13 天", "product_launch_plan.md", "时间线", ["2026-09-25", "2026-10-08"], []),
        ("Memory V2 如何同时保留历史又控制 prompt 长度？", "每 20 条消息冻结 segment 并维护 running summary，原始消息保留但不全部进入上下文。", "memory_design_v2.md", "对话压缩", ["20 条消息", "running summary", "不会全部进入模型上下文"], []),
        ("混合检索全量切换需要同时满足哪两个数值条件？", "MRR@10 至少 0.75，且无答案误召回率不高于 10%。", "chinese_risk_notes_v2.md", "决策", ["0.75", "10%"], []),
    ]
    for question, answer, filename, heading, facts, negatives in parent_specs:
        cases.append(case("parent_context", question, answer, [filename], evidence(filename, heading, *facts), negatives=negatives, tags=["multi_fact"]))

    conflict_specs = [
        ("当前生效合同的退款期限是多少？", "30 天", "contract_2026_final.md", "付款与退款", ["30 天"], ["contract_2024_archive.md", "contract_2025_revision.md", "contract_2026_draft.md"]),
        ("当前生效合同约定由哪个仲裁机构处理？", "上海仲裁委员会", "contract_2026_final.md", "争议处理", ["上海仲裁委员会"], ["contract_2024_archive.md", "contract_2025_revision.md", "contract_2026_draft.md"]),
        ("2026 最终合同相比草案总额增加了多少？", "8,000 元", "contract_2026_final.md", "合同信息", ["128,000"], ["contract_2026_draft.md"]),
        ("当前 RAG 方案使用固定 500-token 切分吗？", "不是；当前使用 Parent 800 tokens、Child 180 tokens。", "rag_architecture_v2.md", "Parent-Child Chunking", ["800 tokens", "180 tokens"], ["rag_architecture_v1.md"]),
        ("当前退款政策对硬件退货收取多少整备费？", "5%", "refund_policy_2026.md", "手续费", ["5%"], ["refund_policy_2024.md"]),
        ("当前数字服务退款期限比旧政策延长多少天？", "16 天", "refund_policy_2026.md", "时限", ["30 天"], ["refund_policy_2024.md"]),
        ("当前 Memory 方案还会把最近 50 轮全部放入 prompt 吗？", "不会；当前通过 segment、running summary 和 GSSC 选择上下文。", "memory_design_v2.md", "上下文选择与对话压缩", ["GSSC", "running summary", "不会全部进入模型上下文"], ["memory_design_v1.md"]),
        ("NX-100 在南仓当前价与 2025 归档价相差多少？", "500", "inventory_south.csv", "row:NX-100", ["8799"], ["pricing_archive_2025.csv"]),
        ("NX-300 在南仓当前有货吗，北仓呢？", "南仓有货 5 件，北仓缺货 0 件。", "inventory_south.csv", "row:NX-300", ["5"], ["inventory_north.csv"]),
        ("当前生产 RAG 后端是 python_bm25 还是 qdrant_hybrid？", "qdrant_hybrid", "rag_architecture_v2.md", "检索链路", ["qdrant_hybrid"], ["rag_architecture_v1.md"]),
    ]
    for question, answer, filename, heading, facts, negatives in conflict_specs:
        docs = [filename]
        gold = evidence(filename, heading, *facts)
        if question.startswith("2026 最终合同相比"):
            docs.append("contract_2026_draft.md")
            gold += evidence("contract_2026_draft.md", "合同信息", "120,000")
        if question.startswith("当前数字服务退款期限比"):
            docs.append("refund_policy_2024.md")
            gold += evidence("refund_policy_2024.md", "时限", "14 天")
        if question.startswith("NX-100"):
            docs.append("pricing_archive_2025.csv")
            gold += evidence("pricing_archive_2025.csv", "row:NX-100", "8299")
        if question.startswith("NX-300"):
            docs.append("inventory_north.csv")
            gold += evidence("inventory_north.csv", "row:NX-300", "0")
        cases.append(case("conflict", question, answer, docs, gold, negatives=negatives, tags=["versioning", "hard_negative"]))

    summary_specs = [
        ("概括 RAG Architecture V2 的核心设计。", "通过 dense+sparse Hybrid Search 和 RRF 排序召回 Child，再回查 Parent 补充上下文，并提供显式 BM25 降级。", "rag_architecture_v2.md", ["dense embedding", "sparse lexical retrieval", "RRF", "parent_id", "python_bm25"]),
        ("概括工具安全策略解决的主要问题。", "按 L0-L4 管理工具风险，对外部写操作审批、危险操作阻断，并以幂等和 checkpoint 保证可审计恢复。", "tool_security_policy.md", ["L0", "L4", "人工审批", "idempotency_key", "checkpoint"]),
        ("总结 Qdrant 事故的原因和改进措施。", "部分重新摄入导致 sparse vector 缺失；随后完成全量重摄入并增加向量一致性和 readiness 检查。", "incident_qdrant_2026_02.md", ["部分重新摄入", "sparse vector", "全量重新摄入", "readiness"]),
        ("总结 PostgreSQL 事故及修复。", "SSE 异常分支泄漏 Session 导致连接池耗尽；修复为统一上下文管理、增加告警和断连测试。", "incident_postgres_2026_03.md", ["没有归还数据库 Session", "context manager", "连接池使用率告警"]),
        ("总结 2026 当前退款政策。", "数字服务未交付部分 30 天内可退且免手续费；硬件 15 天内可退，非质量问题收 5% 整备费。", "refund_policy_2026.md", ["30 天", "15 天", "不收手续费", "5%"]),
        ("总结 Orion 发布计划的时间、目标和风险。", "9 月内测和代码冻结、10 月 8 日公开发布；目标 2000 个团队和 97% 工作流成功率，风险包括移动审批、成本和英文召回。", "product_launch_plan.md", ["2026-09-10", "2026-09-25", "2026-10-08", "2,000", "97%", "移动端审批"]),
        ("概括 Memory V2 如何管理长期对话。", "用三层记忆区分状态、经历与稳定事实，通过 GSSC、segment 和 running summary 控制上下文。", "memory_design_v2.md", ["Working Memory", "Episodic Memory", "Semantic Memory", "GSSC", "running summary"]),
        ("总结混合检索上线的三类风险。", "风险包括 sparse vector 缺失、中文短查询产生噪声、Parent 回查失败。", "chinese_risk_notes_v2.md", ["sparse vector 缺失", "中文短查询", "Parent 回查失败"]),
        ("概括部署手册的发布与回滚原则。", "数据库迁移、API、worker 按顺序发布；应用可回滚，但数据库降级必须单独审核。", "deployment_runbook.md", ["alembic upgrade head", "FastAPI API", "worker", "单独审核"]),
        ("概括员工差旅政策。", "短途优先高铁，住宿按城市限额，超标需批准，结束后 10 个工作日内提交报销。", "employee_travel_policy.md", ["5 小时", "650 元", "450 元", "直属经理书面批准", "10 个工作日"]),
    ]
    for question, answer, filename, facts in summary_specs:
        cases.append(case("summary", question, answer, [filename], evidence(filename, "document", *facts), tags=["long_answer"]))

    unanswerable_specs = [
        "2026 最终合同由哪位见证人签字？",
        "2026 最终合同适用哪个国家的法律？",
        "RAG V2 使用的 dense embedding 模型具体叫什么？",
        "部署手册要求使用哪个 Kubernetes namespace？",
        "L3 审批人的真实姓名是什么？",
        "北仓 NX-100 的产品颜色是什么？",
        "Qdrant 事故造成了多少人民币损失？",
        "2026 退款政策的客服电话是多少？",
        "Orion 产品的最终销售价格是多少？",
        "Memory V2 使用的向量维度是多少？",
    ]
    for question in unanswerable_specs:
        cases.append(case("unanswerable", question, None, [], [], tags=["abstention"]))

    _assign_ids_and_splits(cases)
    return cases


def _assign_ids_and_splits(cases: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in cases:
        grouped.setdefault(item["category"], []).append(item)
    for category, items in grouped.items():
        test_count = CATEGORY_TEST_COUNTS[category]
        split_at = len(items) - test_count
        for index, item in enumerate(items, start=1):
            item["id"] = f"{category}_{index:03d}"
            item["split"] = "dev" if index <= split_at else "test"
            item["schema_version"] = 2


def validate_dataset(cases: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [item["id"] for item in cases]
    if len(ids) != len(set(ids)):
        errors.append("case ids are not unique")
    if len(DOCUMENTS) != 22:
        errors.append(f"expected 22 documents, got {len(DOCUMENTS)}")
    if len(cases) != 100:
        errors.append(f"expected 100 cases, got {len(cases)}")

    category_counts = Counter(item["category"] for item in cases)
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        errors.append(f"unexpected category counts: {dict(category_counts)}")
    split_counts = Counter(item["split"] for item in cases)
    if split_counts != {"dev": 70, "test": 30}:
        errors.append(f"unexpected split counts: {dict(split_counts)}")

    for item in cases:
        for filename in item["relevant_documents"] + item["hard_negative_documents"]:
            if filename not in DOCUMENTS:
                errors.append(f"{item['id']}: unknown document {filename}")
        if item["answerable"] and not item["gold_evidence"]:
            errors.append(f"{item['id']}: answerable case has no gold evidence")
        if not item["answerable"] and (item["gold_answer"] is not None or item["gold_evidence"]):
            errors.append(f"{item['id']}: unanswerable case contains answer/evidence")
        for gold in item["gold_evidence"]:
            filename = gold["filename"]
            content = DOCUMENTS.get(filename, "")
            for fact in gold["required_facts"]:
                if str(fact).lower() not in content.lower():
                    errors.append(f"{item['id']}: fact {fact!r} not found in {filename}")

    if errors:
        raise ValueError("\n".join(errors))
    return {
        "documents": len(DOCUMENTS),
        "cases": len(cases),
        "category_counts": dict(category_counts),
        "split_counts": dict(split_counts),
    }


def write_dataset() -> dict[str, Any]:
    cases = build_cases()
    summary = validate_dataset(cases)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in DOCUMENTS.items():
        (CORPUS_DIR / filename).write_text(content.strip() + "\n", encoding="utf-8")

    for split in ("dev", "test"):
        path = DATASET_DIR / f"cases.{split}.jsonl"
        rows = [item for item in cases if item["split"] == split]
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")

    checksums = {
        filename: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for filename, content in sorted(DOCUMENTS.items())
    }
    manifest = {
        "name": "rag_eval_v2",
        "schema_version": 2,
        "description": "Offline synthetic RAG gold set with hard negatives and frozen test split.",
        **summary,
        "document_sha256": checksums,
    }
    (DATASET_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    manifest = write_dataset()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
