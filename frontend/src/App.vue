<script setup lang="ts">
import { ref } from 'vue'

const loading = ref(false)
const result = ref('还没有请求后端')

async function checkBackend() {
  loading.value = true
  result.value = '正在请求 FastAPI...'

  try {
    const response = await fetch('http://127.0.0.1:8000/api/health')

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const data = await response.json()
    result.value = `${data.status}: ${data.message}`
  } catch (error) {
    result.value = `请求失败：${error}`
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <main class="page">
    <section class="card">
      <h1>Open Deep Research Web</h1>
      <p>Vue 前端已经启动，现在测试连接 FastAPI 后端。</p>

      <button :disabled="loading" @click="checkBackend">
        {{ loading ? '请求中...' : '测试 FastAPI /api/health' }}
      </button>

      <pre>{{ result }}</pre>
    </section>
  </main>
</template>

<style scoped>
.page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f6f7f9;
  font-family:
    Inter,
    system-ui,
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    sans-serif;
}

.card {
  width: 520px;
  padding: 32px;
  border-radius: 16px;
  background: white;
  box-shadow: 0 12px 40px rgba(15, 23, 42, 0.08);
}

h1 {
  margin: 0 0 12px;
  font-size: 28px;
}

p {
  color: #555;
  line-height: 1.7;
}

button {
  margin-top: 20px;
  padding: 10px 18px;
  border: none;
  border-radius: 8px;
  background: #10b981;
  color: white;
  cursor: pointer;
  font-size: 15px;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}

pre {
  margin-top: 20px;
  padding: 16px;
  border-radius: 8px;
  background: #111827;
  color: #d1fae5;
  white-space: pre-wrap;
}
</style>
