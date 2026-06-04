import { useEffect, useState } from 'react'

import { me } from '../api/auth'
import type { CurrentUser } from '../api/types'
import { ErrorState } from '../components/common/ErrorState'
import { JsonBlock } from '../components/common/JsonBlock'
import { LoadingState } from '../components/common/LoadingState'
import { PageHeader } from '../components/common/PageHeader'

export function ProfilePage() {
  const [user, setUser] = useState<CurrentUser | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { me().then(setUser).catch((exc) => setError(exc.message)) }, [])
  return <section className="workbench-page"><PageHeader title="个人资料" description="当前登录用户信息。" />{error ? <ErrorState message={error} /> : user ? <div className="panel"><h2>{user.email}</h2><p className="muted">昵称：{user.nickname || '未设置'}</p><details><summary>技术详情</summary><JsonBlock value={user} /></details></div> : <LoadingState />}</section>
}
