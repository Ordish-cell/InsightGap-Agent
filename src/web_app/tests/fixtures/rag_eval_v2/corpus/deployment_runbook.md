# Agent 平台部署手册

## 健康检查

HTTP 健康检查地址为 `/api/v1/health`。服务正常时返回 `status=ok`。

## 发布顺序

先执行 `alembic upgrade head`，再启动 FastAPI API，最后启动 worker。前端构建命令是 `npm run build`。

## 回滚

应用回滚命令为 `deployctl rollback --service agent-api --to previous`。数据库迁移默认不自动降级，必须执行单独审核。

## 性能目标

预热后的检索 P95 目标为 2.5 秒，首个 SSE 事件目标为 800 毫秒以内。
