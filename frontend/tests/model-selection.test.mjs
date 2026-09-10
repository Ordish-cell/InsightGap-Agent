import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import ts from 'typescript'
const source = readFileSync(new URL('../src/components/llm/modelUi.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText
const { availableModels, readableError, connectionStatus } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`)
const connection = (id, name, status = 'active', last_test_status = 'passed') => ({ id, display_name: name, provider: 'custom', status, last_test_status, models: [{ id: id * 10, model_id: 'text-v1', display_name: 'Text', enabled: true }, { id: id * 10 + 1, model_id: 'off', display_name: 'Off', enabled: false }] })
test('search combines provider, connection and model without returning disabled models', () => {
  const connections = [connection(1, 'Work'), connection(2, 'Personal'), connection(3, 'Draft', 'draft', 'untested')]
  assert.equal(availableModels(connections).length, 2)
  assert.equal(availableModels(connections, ' personal ')[0].connection.id, 2)
  assert.equal(availableModels(connections, 'text-v1').length, 2)
  assert.equal(availableModels(connections, 'off').length, 0)
})
test('failed tests never show the connection as verified', () => {
  assert.equal(availableModels([connection(1, 'Work', 'active', 'failed')]).length, 0)
  assert.equal(connectionStatus(connection(1, 'Work', 'active', 'failed')), '测试失败')
})
test('setup and provider failures explain a recoverable next step', () => {
  assert.match(readableError(new Error('MODEL_CREDENTIALS_ENCRYPTION_KEY is required')), /后端/)
  assert.match(readableError(new Error('authentication_failed')), /API Key/)
  assert.match(readableError(new Error('connection_changed_during_test')), /重新测试/)
})
