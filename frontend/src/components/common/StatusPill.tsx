import { relationLabel, safetyLevelLabel, statusLabel } from '../../utils/labels'

export function StatusPill({ value }: { value?: string }) {
  const text = value || 'unknown'
  const label = statusLabel(text) !== text ? statusLabel(text) : safetyLevelLabel(text) !== text ? safetyLevelLabel(text) : relationLabel(text)
  return <span className={`status-pill ${text.toLowerCase().replaceAll('_', '-')}`}>{label}</span>
}
