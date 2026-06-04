import { useEffect, useRef, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'

import { me } from '../../api/auth'
import type { CurrentUser } from '../../api/types'

type NavItem = { label: string; short: string; href: string; icon: string }

const links: NavItem[] = [
  { label: '首页', short: '首页', href: '/', icon: '⌂' },
  { label: '信息流', short: '信息', href: '/feed', icon: '◆' },
  { label: '深度研究', short: '研究', href: '/research', icon: '◉' },
  { label: '成果库', short: '成果', href: '/artifacts', icon: '▣' },
  { label: '长期记忆', short: '记忆', href: '/memory', icon: '◎' },
  { label: '技能库', short: '技能', href: '/skills', icon: '✓' },
  { label: '审批台', short: '审批', href: '/approvals', icon: '!' },
  { label: '工具审计', short: '工具', href: '/mcp', icon: '◌' },
]

export function Sidebar() {
  const navigate = useNavigate()
  const menuRef = useRef<HTMLDivElement | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const [user, setUser] = useState<CurrentUser | null>(null)
  const [expanded, setExpanded] = useState(() => localStorage.getItem('sidebarExpanded') !== 'false')

  useEffect(() => {
    localStorage.setItem('sidebarExpanded', String(expanded))
  }, [expanded])
  useEffect(() => {
    me().then(setUser).catch(() => setUser(null))
  }, [])
  useEffect(() => {
    function close(event: MouseEvent) {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false)
    }
    window.addEventListener('mousedown', close)
    return () => window.removeEventListener('mousedown', close)
  }, [])

  function logout() {
    localStorage.removeItem('authToken')
    navigate('/login', { replace: true })
  }

  return (
    <aside className={expanded ? 'sidebar expanded' : 'sidebar collapsed'}>
      <div className="sidebar-top">
        <button className="sidebar-toggle" onClick={() => setExpanded((value) => !value)} aria-label="展开或收起侧边栏">
          <span className="sidebar-toggle-icon">{expanded ? '‹' : '›'}</span>
        </button>
        <NavLink to="/" className="brand">
          <span className="brand-mark">OS</span>
          {expanded ? (
            <span className="brand-text">
              <strong>信息差 Agent OS</strong>
              <small>Gap Intelligence Workbench</small>
            </span>
          ) : null}
        </NavLink>
      </div>

      <nav className="sidebar-nav">
        {links.map((item) => (
          <NavLink key={item.href} to={item.href} end={item.href === '/'} className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}>
            <span className="nav-icon">{item.icon}</span>
            {expanded ? (
              <>
                <span className="nav-label">{item.label}</span>
                <span className="nav-short">{item.short}</span>
              </>
            ) : null}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-bottom" ref={menuRef}>
        {menuOpen ? (
          <div className="account-popover">
            <div className="account-popover-user">
              <span className="account-avatar">OS</span>
              <div>
                <strong>{user?.email || '已登录用户'}</strong>
                <small>当前工作空间：MX87</small>
              </div>
            </div>
            <div className="account-popover-line" />
            <NavLink className="account-menu-item" to="/profile" onClick={() => setMenuOpen(false)}>
              <span>◉</span>
              <strong>个人资料</strong>
            </NavLink>
            <NavLink className="account-menu-item" to="/settings" onClick={() => setMenuOpen(false)}>
              <span>⚙</span>
              <strong>设置</strong>
              <em>Ctrl ,</em>
            </NavLink>
            <NavLink className="account-menu-item" to="/agent" onClick={() => setMenuOpen(false)}>
              <span>⌁</span>
              <strong>Agent 调试</strong>
            </NavLink>
            <div className="account-popover-line" />
            <button className="account-menu-item danger" onClick={logout}>
              <span>→</span>
              <strong>退出登录</strong>
            </button>
          </div>
        ) : null}
        <button className="sidebar-settings-trigger" onClick={() => setMenuOpen((value) => !value)}>
          <span className="nav-icon">⚙</span>
          {expanded ? <span>设置</span> : null}
        </button>
      </div>
    </aside>
  )
}
