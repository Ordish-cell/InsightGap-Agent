# Memory 设计 V2（当前方案）

状态：active。

## 三层记忆

Working Memory 保存当前任务临时状态；Episodic Memory 保存具有时间和行为背景的经历；Semantic Memory 保存稳定偏好与事实。

## 上下文选择

GSSC 按 Gather、Select、Structure、Compress 四阶段组装上下文。候选内容按相关性、重要性和 token budget 选择，不把所有记忆直接塞入 prompt。

## 对话压缩

每 20 条消息冻结一个历史 segment，同时维护 running summary。原始消息仍留在会话存储中，但不会全部进入模型上下文。
