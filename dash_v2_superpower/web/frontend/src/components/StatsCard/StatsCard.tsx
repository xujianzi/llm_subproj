import { useMapStore } from '../../store/useMapStore'

function fmt(n: number) {
  return n >= 1000 ? n.toLocaleString('en-US', { maximumFractionDigits: 0 }) : n.toFixed(2)
}

export function StatsCard() {
  const stats = useMapStore(s => s.stats)
  if (!stats) return null

  const rows = [
    { label: 'Min',    value: fmt(stats.min) },
    { label: 'Max',    value: fmt(stats.max) },
    { label: 'Mean',   value: fmt(stats.mean) },
    { label: 'Median', value: fmt(stats.median) },
  ]

  return (
    <div className="absolute bottom-8 left-4 z-10 w-48
      bg-panel/80 backdrop-blur border border-border rounded-xl p-3 shadow-xl">
      <p className="text-xs text-muted mb-2 truncate">{stats.variable}</p>
      {rows.map(r => (
        <div key={r.label} className="flex justify-between text-xs mb-1">
          <span className="text-muted">{r.label}</span>
          <span className="text-text font-mono">{r.value}</span>
        </div>
      ))}
    </div>
  )
}
