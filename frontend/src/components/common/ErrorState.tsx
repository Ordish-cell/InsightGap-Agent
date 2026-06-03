export function ErrorState({ title = 'Request failed', message, action }: { title?: string; message: string; action?: React.ReactNode }) {
  return (
    <div className="error-state">
      <strong>{title}</strong>
      <p>{message}</p>
      {action ? <div>{action}</div> : null}
    </div>
  )
}
