import { useEffect, useRef } from 'react'
import { Outlet, useLocation } from 'react-router-dom'

import { Sidebar } from './Sidebar'

export function AppShell() {
  const location = useLocation()
  const isHome = location.pathname === '/'
  const content = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      content.current?.animate([{ opacity: .4, transform: 'translateY(4px)' }, { opacity: 1, transform: 'none' }], { duration: 200, easing: 'cubic-bezier(.22,1,.36,1)' })
    }
  }, [location.pathname])

  return (
    <div className="app-shell">
      <Sidebar />
      <main className={isHome ? 'app-main home-main' : 'app-main'}>
        <div ref={content} className={isHome ? 'home-content' : 'content'}>
          <Outlet />
        </div>
      </main>
    </div>
  )
}
