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
  return <><PageHeader title="Profile" description="Current authenticated user." />{error ? <ErrorState message={error} /> : user ? <div className="panel"><JsonBlock value={user} /></div> : <LoadingState />}</>
}
