import { Outlet, useLocation } from 'react-router-dom'

import { Sidebar } from './Sidebar'

export function AppShell() {
  const location = useLocation()
  const isHome = location.pathname === '/'

  return (
    <div className="app-shell">
      <Sidebar />
      <main className={isHome ? 'app-main home-main' : 'app-main'}>
        <div className={isHome ? '' : 'content'}>
          <Outlet />
        </div>
      </main>
    </div>
  )
}
