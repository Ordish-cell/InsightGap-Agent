import { useEffect, useRef, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'

import * as agent from '../../api/agent'
import { me } from '../../api/auth'
import type { AgentConversation, CurrentUser } from '../../api/types'
import { ConfirmModal } from '../common/ConfirmModal'

type NavItem = { label: string; short: string; href: string; icon: string }

const links: NavItem[] = [
  { label: '首页', short: '首页', href: '/', icon: '⌂' },
  { label: '信息流', short: '信息', href: '/feed', icon: '◆' },
  { label: '深度研究', short: '研究', href: '/research', icon: '◎' },
  { label: '成果库', short: '成果', href: '/artifacts', icon: '▣' },
  { label: '长期记忆', short: '记忆', href: '/memory', icon: '◌' },
  { label: '技能库', short: '技能', href: '/skills', icon: '✓' },
  { label: '审批台', short: '审批', href: '/approvals', icon: '!' },
  { label: '工具审计', short: '工具', href: '/mcp', icon: '◈' },
]

export function Sidebar() {
  const navigate = useNavigate()
  const menuRef = useRef<HTMLDivElement | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const [user, setUser] = useState<CurrentUser | null>(null)
  const [expanded, setExpanded] = useState(() => localStorage.getItem('sidebarExpanded') !== 'false')
  const [conversationOpen, setConversationOpen] = useState(false)
  const [conversationLoading, setConversationLoading] = useState(false)
  const [conversations, setConversations] = useState<AgentConversation[]>([])
  const [deleteTarget, setDeleteTarget] = useState<AgentConversation | null>(null)

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

  async function loadConversations() {
    setConversationLoading(true)
    try {
      const result = await agent.listConversations({ status: 'active', limit: 50 })
      setConversations(result.items || [])
    } finally {
      setConversationLoading(false)
    }
  }

  function toggleConversations() {
    const next = !conversationOpen
    setConversationOpen(next)
    if (next) void loadConversations()
  }

  function openConversation(conversationId: string) {
    sessionStorage.setItem('agentOpenConversationId', conversationId)
    navigate('/')
    window.dispatchEvent(new CustomEvent('agent:open-conversation', { detail: { conversationId } }))
  }

  function newConversation() {
    sessionStorage.removeItem('agentOpenConversationId')
    navigate('/')
    window.dispatchEvent(new CustomEvent('agent:new-conversation'))
  }

  async function confirmDeleteConversation() {
    const item = deleteTarget
    if (!item) return
    setDeleteTarget(null)
    const openId = sessionStorage.getItem('agentOpenConversationId')
    const isCurrentConversation = openId === item.conversation_id

    // Optimistic: remove from list immediately
    setConversations((prev) => prev.filter((c) => c.conversation_id !== item.conversation_id))

    if (isCurrentConversation) {
      sessionStorage.removeItem('agentOpenConversationId')
      navigate('/', { replace: true })
      // Small delay so navigation commits before the custom event fires
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('agent:new-conversation'))
      }, 80)
    }

    try {
      await agent.hardDeleteConversation(item.conversation_id)
      await loadConversations()
    } catch {
      // Refresh list on failure to undo optimistic removal
      void loadConversations()
    }
  }

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

        <button className={conversationOpen ? 'nav-link nav-button active' : 'nav-link nav-button'} type="button" onClick={toggleConversations}>
          <span className="nav-icon">☰</span>
          {expanded ? (
            <>
              <span className="nav-label">会话管理</span>
              <span className="nav-short">{conversationOpen ? '收起' : '展开'}</span>
            </>
          ) : null}
        </button>

        {expanded && conversationOpen ? (
          <div className="sidebar-conversation-panel">
            <div className="sidebar-conversation-actions">
              <button type="button" onClick={newConversation}>
                新建会话
              </button>
              <button type="button" onClick={() => void loadConversations()}>
                刷新
              </button>
            </div>
            <div className="sidebar-conversation-list">
              {conversationLoading ? <span className="sidebar-conversation-empty">正在加载会话</span> : null}
              {!conversationLoading && !conversations.length ? <span className="sidebar-conversation-empty">还没有会话</span> : null}
              {conversations.map((item) => (
                <div className="sidebar-conversation-item-row" key={item.conversation_id}>
                  <button className="sidebar-conversation-item" type="button" onClick={() => openConversation(item.conversation_id)}>
                    <strong>{item.title || '未命名会话'}</strong>
                    <span>{item.last_message_preview || '暂无消息'}</span>
                  </button>
                  <button
                    className="sidebar-conversation-delete"
                    type="button"
                    title="彻底删除会话"
                    onClick={(e) => { e.stopPropagation(); setDeleteTarget(item) }}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        ) : null}
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
              <span>◐</span>
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
              <span>↗</span>
              <strong>退出登录</strong>
            </button>
          </div>
        ) : null}
        <button className="sidebar-settings-trigger" onClick={() => setMenuOpen((value) => !value)}>
          <span className="nav-icon">⚙</span>
          {expanded ? <span>设置</span> : null}
        </button>
      </div>

      <ConfirmModal
        open={Boolean(deleteTarget)}
        title="删除会话"
        message={`确定要彻底删除「${deleteTarget?.title || '未命名会话'}」吗？所有对话记录和 Agent 运行数据都会被永久删除，无法恢复。`}
        confirmLabel="彻底删除"
        danger
        onConfirm={confirmDeleteConversation}
        onCancel={() => setDeleteTarget(null)}
      />
    </aside>
  )
}
