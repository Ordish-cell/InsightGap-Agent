import type { LlmConnection } from '../../api/types'
export const protocolLabels: Record<string, string> = {
  openai_chat_completions: 'OpenAI Chat Completions', openai_responses: 'OpenAI Responses',
  anthropic_messages: 'Anthropic Messages', google_generate_content: 'Google Gemini', ollama_chat: 'Ollama',
}
export function isConnectionReady(connection: LlmConnection) {
  return connection.status === 'active' && connection.last_test_status === 'passed'
}
export function connectionStatus(connection: LlmConnection) {
  return isConnectionReady(connection) ? '已验证' : connection.last_test_status === 'failed' ? '测试失败' : '待测试'
}
export function availableModels(connections: LlmConnection[], query = '') {
  const search = query.trim().toLowerCase()
  return connections.filter(isConnectionReady).flatMap(connection => connection.models.filter(model => model.enabled).map(model => ({ connection, model })))
    .filter(({ connection, model }) => `${connection.display_name} ${connection.provider} ${model.display_name} ${model.model_id}`.toLowerCase().includes(search))
}
export function readableError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error)
  const labels: Record<string, string> = {
    MODEL_CREDENTIALS_ENCRYPTION_KEY: '后端尚未配置有效的模型加密密钥，请配置 .env 中的 MODEL_CREDENTIALS_ENCRYPTION_KEY 后重启。',
    'Stored model credentials cannot be decrypted': '无法解密已保存密钥。请恢复原加密密钥，或重新创建连接。',
    authentication_failed: '认证失败，请检查 API Key、认证方式和自定义 Headers。',
    endpoint_or_model_not_found: '地址或模型不存在，请核对 API 协议、地址和模型 ID。',
    connection_timeout: '请求超时，请检查网络、服务地址或代理后重试。',
    connection_unreachable: '无法连接服务，请检查地址及本地服务是否启动。',
    rate_limited: '服务正在限流或配额不足，请稍后再试。',
    model_id_required: '请先选择或输入要测试的模型 ID。',
    protocol_not_supported: '该供应商不支持所选协议。',
    model_discovery_unavailable: '这个接口没有模型目录，请手动添加模型 ID 后测试。',
    connection_changed_during_test: '测试期间配置已变更，请使用最新配置重新测试。',
    empty_model_response: '服务没有返回文本，暂不能确认模型可用于聊天。',
    invalid_url: '地址需要以 http:// 或 https:// 开头，不能包含密钥、查询参数或片段。',
    invalid_custom_headers: 'Headers 的名称和值必须是单行字符串。',
    custom_headers_must_be_an_object: '自定义 Headers 必须是 JSON 对象。',
    missing_required_fields: '请填写必填项后保存。',
    model_not_available: '连接尚未通过测试，或模型已停用。',
  }
  return Object.entries(labels).find(([code]) => message.includes(code))?.[1] || message
}
