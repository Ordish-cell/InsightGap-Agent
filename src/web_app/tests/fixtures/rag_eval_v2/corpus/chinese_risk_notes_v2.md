# 混合检索上线风险说明

## 背景

本说明用于评估 Qdrant Hybrid Search 从灰度切换为默认检索后端的风险。

## 风险

第一，部分重新摄入会造成 sparse vector 缺失，使结果静默偏向 dense。第二，中文短查询可能产生过多二元词，导致无关文档排名升高。第三，Parent 回查失败会让正确 Child 缺少回答上下文。

## 缓解措施

上线前核对 dense/sparse point 数量，按 exact、table、semantic、summary 分桶报告指标，并在 Parent 回查失败时记录 `parent_context_missing` warning。

## 决策

只有冻结测试集 MRR@10 达到 0.75 且无答案误召回率不高于 10% 时，才允许全量切换。
