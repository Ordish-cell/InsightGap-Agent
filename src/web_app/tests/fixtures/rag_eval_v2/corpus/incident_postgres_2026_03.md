# INC-2026-034 PostgreSQL 连接池耗尽复盘

## 影响

2026-03-08 14:20 至 15:43，Agent run 创建接口大量超时，事故持续 83 分钟，严重级别为 SEV-1。

## 根因

SSE 客户端断开后，两个异常分支没有归还数据库 Session，连接池逐渐耗尽。

## 修复

统一使用 context manager 管理 Session，增加连接池使用率告警，并为 SSE 断连路径加入集成测试。

## 负责人

事故负责人为 Wang Lei。
