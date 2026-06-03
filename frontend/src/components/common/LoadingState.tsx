export function LoadingState({ title = 'Loading', compact = false }: { title?: string; compact?: boolean }) {
  return (
    <div className={compact ? 'loading-state compact' : 'loading-state'}>
      <span className="skeleton-dot" />
      <span>{title}</span>
    </div>
  )
}
