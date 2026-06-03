import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/layout/AppShell'
import { AgentRunPage } from './pages/AgentRunPage'
import { ApprovalsPage } from './pages/ApprovalsPage'
import { ArtifactsPage } from './pages/ArtifactsPage'
import { FeedCardDetailPage } from './pages/FeedCardDetailPage'
import { FeedPage } from './pages/FeedPage'
import { LoginPage } from './pages/LoginPage'
import { McpToolCallsPage } from './pages/McpToolCallsPage'
import { MemoryPage } from './pages/MemoryPage'
import { ProfilePage } from './pages/ProfilePage'
import { ResearchRunDetailPage } from './pages/ResearchRunDetailPage'
import { ResearchRunsPage } from './pages/ResearchRunsPage'
import { SettingsPage } from './pages/SettingsPage'
import { SkillsPage } from './pages/SkillsPage'

function RequireAuth({ children }: { children: React.ReactNode }) {
  return localStorage.getItem('authToken') ? children : <Navigate to="/login" replace />
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
        <Route index element={<Navigate to="/feed" replace />} />
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
      <Route path="*" element={<Navigate to="/feed" replace />} />
    </Routes>
  )
}
