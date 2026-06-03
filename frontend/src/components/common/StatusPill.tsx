export function StatusPill({ value }: { value?: string }) {
  const text = value || 'unknown'
  return <span className={`status-pill ${text.toLowerCase().replaceAll('_', '-')}`}>{text}</span>
}
