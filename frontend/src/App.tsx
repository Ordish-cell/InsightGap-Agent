import { useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import * as auth from './api/auth'
import { AppShell } from './components/layout/AppShell'
import { AgentRunPage } from './pages/AgentRunPage'
import { ApprovalsPage } from './pages/ApprovalsPage'
import { ArtifactsPage } from './pages/ArtifactsPage'
import { FeedCardDetailPage } from './pages/FeedCardDetailPage'
import { FeedPage } from './pages/FeedPage'
import { HomePage } from './pages/HomePage'
import { LoginPage } from './pages/LoginPage'
import { McpToolCallsPage } from './pages/McpToolCallsPage'
import { MemoryPage } from './pages/MemoryPage'
import { ProfilePage } from './pages/ProfilePage'
import { ResearchRunDetailPage } from './pages/ResearchRunDetailPage'
import { ResearchRunsPage } from './pages/ResearchRunsPage'
import { SettingsPage } from './pages/SettingsPage'
import { SkillsPage } from './pages/SkillsPage'

function RequireAuth({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<'checking' | 'authenticated' | 'unauthenticated'>(() =>
    localStorage.getItem('authToken') ? 'checking' : 'unauthenticated',
  )

  useEffect(() => {
    let active = true
    const handleUnauthorized = () => {
      if (active) setStatus('unauthenticated')
    }

    window.addEventListener('auth:unauthorized', handleUnauthorized)
    if (!localStorage.getItem('authToken')) {
      setStatus('unauthenticated')
    } else {
      setStatus('checking')
      auth.me()
        .then(() => {
          if (active) setStatus('authenticated')
        })
        .catch(() => {
          localStorage.removeItem('authToken')
          if (active) setStatus('unauthenticated')
        })
    }

    return () => {
      active = false
      window.removeEventListener('auth:unauthorized', handleUnauthorized)
    }
  }, [])

  if (status === 'checking') return <main className="simple-login-page" aria-busy="true">正在验证登录状态...</main>
  return status === 'authenticated' ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        <Route index element={<HomePage />} />
        <Route path="feed" element={<FeedPage />} />
        <Route path="feed/:cardId" element={<FeedCardDetailPage />} />
        <Route path="research" element={<ResearchRunsPage />} />
        <Route path="research/:researchRunId" element={<ResearchRunDetailPage />} />
        <Route path="agent" element={<AgentRunPage />} />
        <Route path="artifacts" element={<ArtifactsPage />} />
        <Route path="memory" element={<MemoryPage />} />
        <Route path="skills" element={<SkillsPage />} />
        <Route path="approvals" element={<ApprovalsPage />} />
        <Route path="mcp" element={<McpToolCallsPage />} />
        <Route path="profile" element={<ProfilePage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
