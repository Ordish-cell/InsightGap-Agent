import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { me } from '../../api/auth'
import { dependencies } from '../../api/health'
import type { CurrentUser, HealthResponse } from '../../api/types'
import { StatusPill } from '../common/StatusPill'

export function Topbar() {
  const navigate = useNavigate()
  const [user, setUser] = useState<CurrentUser | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    me().then(setUser).catch(() => setUser(null))
    dependencies().then(setHealth).catch(() => setHealth(null))
  }, [])

  function logout() {
    localStorage.removeItem('authToken')
    navigate('/login', { replace: true })
  }

  return (
    <header className="topbar">
      <div className="topbar-status">
        <StatusPill value={health?.mysql ? 'api_ok' : 'checking'} />
        <span className="muted">MCP {health?.mcp ? 'ready' : 'pending'}</span>
      </div>
      <div className="topbar-user">
        <span>{user?.email || 'Signed in'}</span>
        <button className="button ghost" onClick={logout}>Log out</button>
      </div>
    </header>
  )
}
