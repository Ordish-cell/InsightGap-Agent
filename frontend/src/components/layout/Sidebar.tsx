import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'

import * as agent from '../../api/agent'
import { me } from '../../api/auth'
import type { AgentConversation, CurrentUser } from '../../api/types'
import { Icon } from '../common/Icon'
import { usePresence } from '../common/motion'
import { ConfirmModal } from '../common/ConfirmModal'

const links = [
  { label: '对话', href: '/', icon: 'chat' },
  { label: '信息流', href: '/feed', icon: 'feed' },
  { label: '深度研究', href: '/research', icon: 'research' },
  { label: '成果库', href: '/artifacts', icon: 'artifact' },
  { label: '长期记忆', href: '/memory', icon: 'memory' },
  { label: '技能库', href: '/skills', icon: 'skill' },
  { label: '审批台', href: '/approvals', icon: 'approval' },
]

export function Sidebar() {
  const navigate = useNavigate()
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)
  const mobilePresent = usePresence(mobileOpen)
  const [managementOpen, setManagementOpen] = useState(false)
  const navRef = useRef<HTMLDivElement>(null)
  const drawerRef = useRef<HTMLElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const accountTriggerRef = useRef<HTMLButtonElement>(null)
  const [indicator, setIndicator] = useState({ y: 0, height: 40, visible: false })
  useLayoutEffect(() => {
    const active = navRef.current?.querySelector<HTMLElement>('[aria-current="page"]')
    setIndicator({ y: active?.offsetTop || 0, height: active?.offsetHeight || 40, visible: !!active })
  }, [location.pathname])
  useEffect(() => { setMobileOpen(false) }, [location.pathname])
  useEffect(() => {
    if (!mobileOpen) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    drawerRef.current?.querySelector<HTMLElement>('a, button')?.focus()
    function key(event: KeyboardEvent) {
      if (event.key === 'Escape') setMobileOpen(false)
      if (event.key === 'Tab') {
        const items = Array.from(drawerRef.current?.querySelectorAll<HTMLElement>('a, button:not(:disabled), summary') || []).filter(el => el.getClientRects().length)
        const first = items[0], last = items[items.length - 1]
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
        if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
      }
    }
    window.addEventListener('keydown', key)
    return () => { document.body.style.overflow = previousOverflow; window.removeEventListener('keydown', key); triggerRef.current?.focus() }
  }, [mobileOpen])
  const menuRef = useRef<HTMLDivElement | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const menuPresent = usePresence(menuOpen)
  const [user, setUser] = useState<CurrentUser | null>(null)
  const [expanded, setExpanded] = useState(() => localStorage.getItem('sidebarExpanded') !== 'false')
  const [conversationOpen, setConversationOpen] = useState(false)
  const [conversationLoading, setConversationLoading] = useState(false)
  const [conversations, setConversations] = useState<AgentConversation[]>([])
  const [deleteTarget, setDeleteTarget] = useState<AgentConversation | null>(null)

  const [deletions, setDeletions] = useState<agent.DeletionTask[]>([])
  const [deletionError, setDeletionError] = useState('')
  async function loadDeletions() {
    try {
      const jobs = await agent.listDeletionTasks()
      setDeletions(jobs)
      const current = sessionStorage.getItem('agentOpenConversationId')
      if (jobs.some(j => j.conversation_id === current && j.status.startsWith('completed'))) newConversation()
    } catch { /* Migration errors are shown when an explicit delete is attempted. */ }
  }
  useEffect(() => {
    void loadDeletions()
    const timer = window.setInterval(() => void loadDeletions(), 2000)
    return () => window.clearInterval(timer)
  }, [])

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
    function key(event: KeyboardEvent) {
      if (event.key === 'Escape' && menuOpen) { setMenuOpen(false); accountTriggerRef.current?.focus() }
    }
    if (menuOpen) menuRef.current?.querySelector<HTMLElement>('.account-popover a')?.focus()
    window.addEventListener('keydown', key)
    window.addEventListener('mousedown', close)
    return () => { window.removeEventListener('mousedown', close); window.removeEventListener('keydown', key) }
  }, [menuOpen])

  async function loadConversations() {
    setConversationLoading(true)
    try {
      const result = await agent.listConversations({ status: 'active', limit: 50 })
      setConversations(result.items || [])
    } catch (exc) {
      setDeletionError(exc instanceof Error ? exc.message : '会话加载失败')
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
    setMobileOpen(false)
    sessionStorage.setItem('agentOpenConversationId', conversationId)
    navigate('/', { replace: true })
    window.dispatchEvent(new CustomEvent('agent:open-conversation', { detail: { conversationId } }))
  }

  function newConversation() {
    setMobileOpen(false)
    sessionStorage.removeItem('agentOpenConversationId')
    navigate('/', { replace: true })
    window.dispatchEvent(new CustomEvent('agent:new-conversation'))
  }

  async function confirmDeleteConversation() {
    const item = deleteTarget
    if (!item) return
    setDeleteTarget(null)
    setDeletionError('')

    try {
      await agent.hardDeleteConversation(item.conversation_id)
      await loadDeletions()
      await loadConversations()
    } catch (exc) {
      const status = (exc as { status?: number }).status
      const msg = exc instanceof Error ? exc.message : String(exc)
      const code = (exc as { details?: { error?: { code?: string } } }).details?.error?.code
      // CONVERSATION_HAS_PENDING_APPROVAL: prompt user to cancel+delete
      if (status === 409 && (code === 'CONVERSATION_HAS_PENDING_APPROVAL' || msg.includes('CONVERSATION_HAS_PENDING_APPROVAL'))) {
        const confirmed = window.confirm(
          '这个会话还有等待审批的操作。要取消这些操作并删除会话吗？\n\n注意：这会取消所有待审批操作，工具不会执行。'
        )
        if (confirmed) {
          try {
            await agent.hardDeleteConversationCancelPending(item.conversation_id)
            await loadDeletions()
            await loadConversations()
          } catch (error) {
            setDeletionError(error instanceof Error ? error.message : String(error))
            void loadConversations()
          }
        } else {
          // User cancelled — re-add to list
          void loadConversations()
        }
        return
      }
      setDeletionError(msg)
      // Retain the conversation until the server accepts deletion.
      void loadConversations()
    }
  }

  function logout() {
    localStorage.removeItem('authToken')
    navigate('/login', { replace: true })
  }

  return (
    <>
    <header className="mobile-bar"><button ref={triggerRef} className="icon-button" aria-label="打开导航" aria-expanded={mobileOpen} onClick={() => { setExpanded(true); setMobileOpen(true) }}><Icon name="menu" /></button><span>InsightGap</span><button className="icon-button" aria-label="新建会话" onClick={newConversation}><Icon name="plus" /></button></header>
    {mobilePresent && <div className={`sidebar-backdrop ${mobileOpen ? 'is-open' : 'is-closing'}`} onClick={() => setMobileOpen(false)} />}
    <aside ref={drawerRef} className={`sidebar ${expanded ? 'expanded' : 'collapsed'} ${mobileOpen ? 'mobile-open' : ''}`} aria-label="主导航">
      {expanded && (deletionError || deletions.some(j => j.status === 'failed')) && <div aria-live="polite">
        {deletionError && <p role="alert">{deletionError}</p>}
        {deletions.filter(j => j.status === 'failed').map(j => <div key={j.id}>
          <small>会话 {j.conversation_id.slice(0, 8)} · </small>
          <span>删除失败，可重试</span>
          {j.error_message && <p>{j.error_message}</p>}
          <button onClick={() => void agent.retryDeletionTask(j.id).then(loadDeletions).catch(e => setDeletionError(String(e)))}>重试删除</button>
        </div>)}
      </div>}
      <div className="sidebar-top">
        <NavLink to="/" className="brand" title="InsightGap · 信息差 Agent OS"><span className="brand-mark"><Icon name="spark" size={21} /></span>{expanded && <span className="brand-text"><strong>InsightGap</strong><small>信息差 Agent OS</small></span>}</NavLink>
        <button className="sidebar-toggle icon-button" onClick={() => setExpanded(value => !value)} aria-label="展开或收起侧边栏"><Icon name="collapse" size={18} /></button>
        <button className="mobile-close icon-button" onClick={() => setMobileOpen(false)} aria-label="关闭导航"><Icon name="close" /></button>
      </div>
      <button className="new-conversation" onClick={newConversation} title="新建会话"><Icon name="plus" size={18} />{expanded && <span>新建会话</span>}</button>
      <nav className="sidebar-nav">
        {expanded && <span className="nav-section-label">工作空间</span>}
        <div className="primary-navigation" ref={navRef}>
          <span className="nav-active-indicator" style={{ transform: `translateY(${indicator.y}px)`, height: indicator.height, opacity: Number(indicator.visible) }} />
          {links.map(item => <NavLink key={item.href} to={item.href} end={item.href === '/'} title={item.label} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}><Icon name={item.icon} />{expanded && <span className="nav-label">{item.label}</span>}</NavLink>)}
        </div>
        <button className="nav-link nav-button section-toggle" aria-expanded={managementOpen} title="管理" onClick={() => { if (!expanded) setExpanded(true); setManagementOpen(value => !value) }}><Icon name="settings" />{expanded && <><span className="nav-label">管理</span><Icon name="chevron" size={15} style={{ transform: managementOpen ? 'rotate(90deg)' : undefined }} /></>}</button>
        <div className={`nav-collapse ${managementOpen && expanded ? 'is-open' : ''}`} inert={!managementOpen || !expanded}><div>
          <NavLink className="nav-link" to="/mcp"><Icon name="audit" /><span>工具审计</span></NavLink>
          <NavLink className="nav-link" to="/agent"><Icon name="code" /><span>Agent 调试</span></NavLink>
        </div></div>
        <button className="nav-link nav-button section-toggle" type="button" onClick={() => { if (!expanded) setExpanded(true); toggleConversations() }} aria-expanded={conversationOpen} title="会话管理"><Icon name="chat" />{expanded && <><span className="nav-label">最近会话</span><Icon name="chevron" size={15} style={{ transform: conversationOpen ? 'rotate(90deg)' : undefined }} /></>}</button>
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
        {menuPresent ? (
          <div className={`account-popover ${menuOpen ? 'is-open' : 'is-closing'}`} inert={!menuOpen}>
            <div className="account-popover-user">
              <span className="account-avatar">OS</span>
              <div>
                <strong>{user?.email || '已登录用户'}</strong>
                <small>个人账号</small>
              </div>
            </div>
            <div className="account-popover-line" />
            <NavLink className="account-menu-item" to="/profile" onClick={() => setMenuOpen(false)}>
              <Icon name="user" size={18} />
              <strong>个人资料</strong>
            </NavLink>
            <NavLink className="account-menu-item" to="/settings" onClick={() => setMenuOpen(false)}>
              <Icon name="settings" size={18} />
              <strong>设置</strong>

            </NavLink>
            <div className="account-popover-line" />
            <button className="account-menu-item danger" onClick={logout}>
              <Icon name="logout" size={18} />
              <strong>退出登录</strong>
            </button>
          </div>
        ) : null}
        <button ref={accountTriggerRef} className="sidebar-settings-trigger" aria-expanded={menuOpen} aria-label="账号与设置" onClick={() => setMenuOpen((value) => !value)}>
          <span className="account-avatar">{(user?.nickname || user?.email || "I").slice(0, 1).toUpperCase()}</span>
          {expanded ? <><span className="account-caption"><strong>{user?.nickname || "我的账号"}</strong><small>账号与设置</small></span><Icon name="chevron" size={16} /></> : null}
        </button>
      </div>

      <ConfirmModal
        open={Boolean(deleteTarget)}
        title="删除会话"
        message={`确定要彻底删除「${deleteTarget?.title || '未命名会话'}」吗？将永久清理对话、临时记忆、独占附件与对应向量。长期记忆、共享文件和已独立保存的 Skill 保留。`}
        confirmLabel="彻底删除"
        danger
        onConfirm={confirmDeleteConversation}
        onCancel={() => setDeleteTarget(null)}
      />
    </aside>
    </>
  )
}
