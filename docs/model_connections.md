# 模型连接与选择改造

本次目标：本机自用，优先方便地在多个供应商之间切换。保留现有 HTTP/SSE、连接与模型表、加密凭据、会话模型选择和 run 快照；没有新增数据库迁移，没有自动更改已保存连接或密钥。

## 参考与取舍

研究了 [cc-switch](https://github.com/farion1231/cc-switch/tree/f3b18df12007d0fd79fd8ad8d310880664015197) 的以下实现：

- `src/components/providers/ProviderCard.tsx`：已配置供应商、当前状态与操作分离。
- `src/components/providers/forms/ProviderPresetSelector.tsx`：供应商预设与搜索。
- `src/components/providers/forms/shared/ModelInputWithFetch.tsx`：允许手动模型 ID，同时提供在线获取入口。

借鉴其交互组织方式，未复制源码或引入它的 Tauri、代理接管、OAuth、计费统计及 CLI 配置管理。

## 使用方式

1. 设置 → 管理模型连接 → 左侧“＋”。
2. 选择供应商预设，填写连接名称、协议、地址和密钥。自定义 Headers 在高级设置中。
3. 保存连接。首个模型 ID 可选，保存时不调用供应商。
4. 点击“获取模型”，或手动添加模型 ID。目录不可用不妨碍手动添加。
5. 对要用的模型点击“测试生成”，会发送一条简短文本请求，可能产生供应商调用费用。
6. 测试通过后，可以设为新会话默认模型。聊天输入栏可按供应商或模型搜索并选择本次发送的模型。

既有会话的明确模型选择不会因为连接失效而静默切到其他供应商；选择器会提示重新选择。运行中的任务保持原来的模型选择。默认模型与本次发送的模型是两个不同状态。

## 接入修复

- 将目录获取与生成测试分开：测试通过代表所选协议和模型确实返回了内容，GET /models 不再作为生成成功的依据。
- 自定义连接允许先保存再获取模型；不支持目录的协议给出手动添加提示。
- 仅改连接名称或保存相同配置，不再使连接失效；修改地址、协议或凭据后需要重测。
- 生成与发现模型的请求返回后检查连接 revision，避免旧请求验证已修改的配置。
- 测试尚未保存的字段只作为预览，不能激活原连接。
- 首次自动设置默认模型时使用实际通过测试的模型，而非目录里的第一项。
- 修复加密的自定义 Headers 没有进入模型工厂的问题，并同步修复研究模型适配入口的同类传输配置问题；没有更改研究图内部逻辑。
- 在线发现支持自定义认证 Header，统一完整 Endpoint 的处理；刷新目录不再重新启用已停用模型。
- 输入校验和错误提示覆盖缺少加密密钥、认证失败、地址错误、限流、超时、连接失败及配置变更。
- JSON 输入在编辑时保留原文，保存时才解析；避免逐字输入被重新序列化、内容丢失。

## 验证与限制

后端使用临时数据库、假模型与 HTTP mock 验证；浏览器测试全部拦截 API，使用合成连接，不访问真实模型供应商、不修改真实账号。

```powershell
.venv/Scripts/python.exe -m pytest src/web_app/tests/test_llm_registry.py src/web_app/tests/test_chat_control.py -q
cd frontend
node --test tests/chat-control.test.mjs tests/model-selection.test.mjs
npm run build
```

浏览器脚本为 `frontend/tests/model-manager.browser.mjs`，需要可用的 Playwright 与 Edge，以及 Vite 服务（默认 127.0.0.1:5175，可用 MODEL_UI_URL 覆盖）。覆盖供应商切换、默认模型、生成测试请求、错误提示、JSON 输入、草稿保存、390px 窄窗口、Escape/焦点恢复和聊天选择器搜索切换。

测试生成仅验证一次文本生成，不认证工具调用、结构化输出、视觉或该供应商所有模型的能力。连接验证状态沿用现有 schema；已有历史测试状态不自动清空。服务地址或模型本身是否可用仍由实际供应商决定。

本轮未替换预设模型目录，没有使用真实凭据进行付费冒烟。运行环境仍需要前一轮的数据库迁移和有效的 MODEL_CREDENTIALS_ENCRYPTION_KEY；本次没有自动补迁移或改动 .env。
