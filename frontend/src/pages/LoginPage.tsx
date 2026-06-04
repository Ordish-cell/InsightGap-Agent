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
    if (localStorage.getItem('authToken')) navigate('/', { replace: true })
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
    <main className="simple-login-page">
      <div className="login-glow glow-one" />
      <div className="login-glow glow-two" />
      <section className="simple-login-card" data-mode={mode}>
        <div className="simple-login-brand">
          <span className="simple-login-logo">OS</span>
          <div><strong>信息差 Agent OS</strong><small>{mode === 'login' ? '登录' : '注册'}后进入工作台</small></div>
        </div>

        <div className="simple-login-tabs" role="tablist" aria-label="登录注册切换">
          <span className="simple-login-tab-indicator" />
          <button className={mode === 'login' ? 'active' : ''} type="button" onClick={() => setMode('login')}>登录</button>
          <button className={mode === 'register' ? 'active' : ''} type="button" onClick={() => setMode('register')}>注册</button>
        </div>

        <form className="simple-login-form" onSubmit={submit}>
          {error ? <ErrorState message={error} /> : null}
          <label>邮箱<input className="input" value={email} onChange={(event) => setEmail(event.target.value)} type="email" placeholder="输入邮箱" required /></label>
          <label>密码<input className="input" value={password} onChange={(event) => setPassword(event.target.value)} type="password" placeholder="输入密码" required /></label>
          <div className={mode === 'register' ? 'nickname-field open' : 'nickname-field'}>
            <label>昵称<input className="input" value={nickname} onChange={(event) => setNickname(event.target.value)} placeholder="可选" /></label>
          </div>
          <button className={canSubmit ? 'simple-login-submit active' : 'simple-login-submit'} disabled={!canSubmit}>{loading ? '处理中...' : mode === 'login' ? '登录' : '注册'}</button>
        </form>
      </section>
    </main>
  )
}
