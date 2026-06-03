import { FormEvent, useEffect, useState } from 'react'
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

  useEffect(() => {
    if (localStorage.getItem('authToken')) navigate('/feed', { replace: true })
  }, [navigate])

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (mode === 'register') await auth.register({ email, password, nickname })
      const result = await auth.login({ email, password })
      localStorage.setItem('authToken', result.access_token)
      navigate('/feed', { replace: true })
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Authentication failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="login-page">
      <form className="login-card card" onSubmit={submit}>
        <div>
          <h1>Information Gap Agent OS</h1>
          <p className="muted">Sign in to turn FeedCards into research, artifacts, memory, and skills.</p>
        </div>
        {error ? <ErrorState message={error} /> : null}
        <label>Email<input className="input" value={email} onChange={(event) => setEmail(event.target.value)} type="email" required /></label>
        <label>Password<input className="input" value={password} onChange={(event) => setPassword(event.target.value)} type="password" required /></label>
        {mode === 'register' ? <label>Nickname<input className="input" value={nickname} onChange={(event) => setNickname(event.target.value)} /></label> : null}
        <button className="button" disabled={loading}>{loading ? 'Working' : mode === 'login' ? 'Log in' : 'Create account'}</button>
        <button className="button ghost" type="button" onClick={() => setMode(mode === 'login' ? 'register' : 'login')}>
          {mode === 'login' ? 'Create an account' : 'Use existing account'}
        </button>
      </form>
    </main>
  )
}
