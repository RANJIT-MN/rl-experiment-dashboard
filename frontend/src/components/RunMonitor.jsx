import { useEffect, useMemo, useState } from 'react'
import { getRunMetrics } from '../services/api'

function MetricChart({ title, points, emptyMessage }) {
  if (points.length === 0) {
    return (
      <div className="run-metric-chart">
        <h5>{title}</h5>
        <p className="chart-empty">{emptyMessage}</p>
      </div>
    )
  }

  const width = 640
  const height = 170
  const padding = { top: 12, right: 16, bottom: 30, left: 48 }
  const values = points.map((point) => point.value)
  const minValue = Math.min(...values)
  const maxValue = Math.max(...values)
  const valueRange = maxValue - minValue || 1
  const minStep = points[0].timestep
  const maxStep = points[points.length - 1].timestep
  const stepRange = maxStep - minStep || 1
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom
  const coordinates = points.map((point) => ({
    x: padding.left + ((point.timestep - minStep) / stepRange) * chartWidth,
    y: padding.top + chartHeight - ((point.value - minValue) / valueRange) * chartHeight,
  }))

  return (
    <div className="run-metric-chart">
      <h5>{title}</h5>
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <line
          x1={padding.left}
          x2={width - padding.right}
          y1={height - padding.bottom}
          y2={height - padding.bottom}
          className="chart-gridline"
        />
        {coordinates.length > 1 && (
          <polyline
            fill="none"
            stroke="#71a7ff"
            strokeWidth="2.5"
            strokeLinejoin="round"
            points={coordinates.map(({ x, y }) => `${x},${y}`).join(' ')}
          />
        )}
        {coordinates.map(({ x, y }, index) => (
          <circle key={`${points[index].timestep}-${index}`} cx={x} cy={y} r="3.5" fill="#71a7ff">
            <title>{`Timestep ${points[index].timestep}: ${points[index].value}`}</title>
          </circle>
        ))}
        <text x={padding.left} y={height - 8} className="chart-label">
          {minStep.toLocaleString()} steps
        </text>
        <text x={width - padding.right} y={height - 8} textAnchor="end" className="chart-label">
          {maxStep.toLocaleString()} steps
        </text>
      </svg>
    </div>
  )
}

export default function RunMonitor({ experiment, run }) {
  const [metrics, setMetrics] = useState([])
  const [error, setError] = useState('')
  const active = ['pending', 'running'].includes(run.status)

  useEffect(() => {
    let disposed = false
    let timer

    async function refresh() {
      try {
        const history = await getRunMetrics(experiment.id, run.id)
        if (!disposed) {
          setMetrics(history.slice(-2000))
          setError('')
        }
      } catch (requestError) {
        if (!disposed) setError(requestError.message)
      }
      if (!disposed && active) timer = window.setTimeout(refresh, 1500)
    }

    refresh()
    return () => {
      disposed = true
      window.clearTimeout(timer)
    }
  }, [experiment.id, run.id, active])

  const latest = metrics[metrics.length - 1] ?? null
  const rewardPoints = useMemo(() => metrics
    .filter((metric) => metric.episode_reward !== null)
    .map((metric) => ({ timestep: metric.global_timestep, value: metric.episode_reward })), [metrics])
  const lengthPoints = useMemo(() => metrics
    .filter((metric) => metric.episode_length !== null)
    .map((metric) => ({ timestep: metric.global_timestep, value: metric.episode_length })), [metrics])
  const progress = Math.min(
    100,
    (latest?.global_timestep ?? 0) / run.requested_timesteps * 100,
  )

  return (
    <div className="run-monitor">
      <div className="run-progress-labels">
        <span>Observed progress</span>
        <strong>{(latest?.global_timestep ?? 0).toLocaleString()} / {run.requested_timesteps.toLocaleString()} steps ({progress.toFixed(1)}%)</strong>
      </div>
      <div className="progress-track" role="progressbar" aria-valuenow={Math.round(progress)} aria-valuemin="0" aria-valuemax="100">
        <span style={{ width: `${progress}%` }} />
      </div>
      {error && <p className="monitor-error" role="status">Run metrics could not be loaded: {error}</p>}
      <div className="run-metric-summary">
        <span>{latest?.completed_episode_count ?? 0} completed episodes</span>
        <span>Latest reward: {latest?.episode_reward == null ? 'Not recorded' : latest.episode_reward.toFixed(2)}</span>
        <span>Speed: {latest?.timesteps_per_second == null ? 'Not recorded' : `${latest.timesteps_per_second.toFixed(1)} steps/s`}</span>
      </div>
      <div className="run-metric-charts">
        <MetricChart title="Training reward by timestep" points={rewardPoints} emptyMessage="No completed training episodes yet." />
        <MetricChart title="Training episode length by timestep" points={lengthPoints} emptyMessage="No completed training episodes yet." />
      </div>
    </div>
  )
}
