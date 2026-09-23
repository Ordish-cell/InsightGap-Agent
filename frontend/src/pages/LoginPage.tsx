import { Icon } from '../components/common/Icon'
import { Tabs } from '../components/common/Tabs'
import { FormEvent, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import * as auth from '../api/auth'
import { ErrorState } from '../components/common/ErrorState'

export function LoginPage() {
  const navigate = useNavigate()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [nickname, setNickname] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const canSubmit = useMemo(() => Boolean(email.trim() && password.trim() && !loading), [email, password, loading])

  useEffect(() => {
    if (!localStorage.getItem('authToken')) return
    let active = true
    auth.me()
      .then(() => {
        if (active) navigate('/', { replace: true })
      })
      .catch(() => localStorage.removeItem('authToken'))
    return () => { active = false }
  }, [navigate])

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!canSubmit) return
    setError('')
    setLoading(true)
    try {
      if (mode === 'register') await auth.register({ email, password, nickname })
      const result = await auth.login({ email, password })
      localStorage.setItem('authToken', result.access_token)
      navigate('/', { replace: true })
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '登录失败，请检查账号或稍后重试')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="login-page">
      <div className="login-glow glow-one" />
      <div className="login-glow glow-two" />
      <section className="login-card" data-mode={mode}>
        <div className="login-brand">
          <span className="login-brand-icon"><Icon name="spark" /></span>
          <div><strong className="login-brand-name">InsightGap</strong><small className="login-brand-sub">{mode === 'login' ? '登录' : '注册'}后进入工作台</small></div>
        </div>

        <Tabs value={mode} onChange={setMode} label="登录注册切换" items={[{ value: 'login', label: '登录' }, { value: 'register', label: '注册' }]} />

        <form className="login-form" onSubmit={submit}>
          {error ? <ErrorState message={error} /> : null}
          <label className="login-field">邮箱<input className="input" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" type="email" placeholder="输入邮箱" required /></label>
          <label className="login-field">密码<input className="input" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} type="password" placeholder="输入密码" required /></label>
          <div inert={mode !== 'register'} className={mode === 'register' ? 'login-nickname-wrap show' : 'login-nickname-wrap'}><div>
            <label className="login-field">昵称<input className="input" value={nickname} onChange={(event) => setNickname(event.target.value)} placeholder="可选" /></label>
          </div></div>
          <button className={canSubmit ? 'login-submit active' : 'login-submit'} disabled={!canSubmit}>{loading ? '处理中...' : mode === 'login' ? '登录' : '注册'}</button>
        </form>
      </section>
    </main>
  )
}
