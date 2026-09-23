export function ActionNotice({ message, error = false }: { message: string; error?: boolean }) {
  if (!message) return null
  return (
    <p className={`action-notice${error ? ' is-error' : ''}`} role={error ? 'alert' : 'status'}>
      {message}
    </p>
  )
}
