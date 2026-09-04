# INC-2026-021 Qdrant 检索异常复盘

## 影响

2026-02-11 10:03 至 10:50，Hybrid 检索出现 502，持续 47 分钟。18% 的查询回退到 BM25，没有发生跨用户数据泄漏。

## 根因

集合 schema 升级后只完成了部分重新摄入，部分 point 缺少 sparse vector；网关健康检查仍然只检查 HTTP 进程存活。

## 修复

完成全量重新摄入，增加 dense/sparse vector 数量一致性检查，并将 collection readiness 加入发布门禁。

## 负责人

事故负责人为 Li Ming，复盘截止日期为 2026-02-14。
