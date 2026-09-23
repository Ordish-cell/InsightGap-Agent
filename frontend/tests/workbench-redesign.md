# 工作台改版验证

浏览器脚本使用 Playwright + Edge，只访问模拟 API 或脚本启动的本地内存服务，不调用真实模型，不修改生产数据。需要可解析的 `playwright` 包与已安装的 Edge；运行前启动 Vite。

```powershell
npm run build
node --test tests/chatStream.test.mjs tests/chat-control.test.mjs tests/live-progress.test.mjs tests/model-selection.test.mjs
$env:WORKBENCH_URL = 'http://127.0.0.1:5175'
$env:CHAT_UI_URL = 'http://127.0.0.1:5175'
$env:MODEL_UI_URL = 'http://127.0.0.1:5175'
node tests/workbench.browser.mjs
node tests/workbench-chat.browser.mjs
node tests/model-manager.browser.mjs
node tests/live-progress.browser.mjs
```

- `workbench.browser.mjs`：13 个工作台路由 × 390/768/1440/1920px；登录、注册；长内容、空数据、加载、失败；反馈、创建研究、防重复提交、成果阅读、技能状态、审批限制、记忆归档恢复、设置保存、抽屉键盘关闭、标签方向键、减少动态效果。
- `workbench-chat.browser.mjs`：附件上传/移除、兼容流式回复、停止、回放、聊天 DOM 与滚动位置保留、手机布局。设置 `CHAT_PROTOCOL=native` 可额外检查原生文本事件。
- 截图保存在系统临时目录的 `insightgap-workbench-redesign`、`insightgap-chat-redesign`、`insight-model-ui`、`insight-live-progress`。

2026-09-23 验证：工作区类型检查、构建、16 项单元测试及上述浏览器脚本通过。现有 `chat-live.browser.mjs` 原生 SSE 回归也在完整工作区通过。构建保留大于 500kB 的主包提示。

独立待提交副本的构建与兼容流式 UI 回归通过。**原生流式回归存在基线问题**：`AgentChatPanel` 的本地助手消息 ID 与服务端消息 ID 不一致时，原生文本投影会过滤事件。工作区开工前已有 `message_id: liveAssistantMessageId` 修复及配套进度改动，已原样保留，未混入本次 UI 提交。因此原生流式通过结果对应完整工作区，不能视为仅本次提交已解决此问题。
