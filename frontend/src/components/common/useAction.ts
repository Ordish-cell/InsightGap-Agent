import { useRef, useState } from 'react'

/** A single in-flight user action, with an explicit result instead of a silent rejection. */
export function useAction() {
  const locked = useRef(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [failed, setFailed] = useState(false)
  async function run(action: () => Promise<unknown>, success = '操作完成。') {
    if (locked.current) return
    locked.current = true
    setBusy(true)
    setMessage('')
    setFailed(false)
    try {
      await action()
      setMessage(success)
    } catch (error) {
      setFailed(true)
      setMessage(error instanceof Error ? error.message : '操作失败，请重试。')
    } finally {
      locked.current = false
      setBusy(false)
    }
  }
  return { busy, message, failed, run }
}
