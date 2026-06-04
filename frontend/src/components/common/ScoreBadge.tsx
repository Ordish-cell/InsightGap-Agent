export function ScoreBadge({ score = 0, label = '评分' }: { score?: number; label?: string }) {
  const level = score >= 0.75 ? 'high' : score >= 0.45 ? 'medium' : 'low'
  return <span className={`score-badge ${level}`}>{label}: {Math.round(score * 100)}</span>
}
