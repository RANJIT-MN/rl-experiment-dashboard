import { useEffect, useMemo, useState } from 'react'
import {
  experimentMetricsWebSocketUrl,
  getExperimentMetrics,
} from '../services/api'

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'stopped'])
const MAX_DISPLAY_METRICS = 2000
const MAX_RECONNECT_ATTEMPTS = 8

function mergeMetrics(current, incoming) {
  const byId = new Map(current.map((metric) => [metric.id, metric]))
  for (const metric of incoming) byId.set(metric.id, metric)
  const recent = [...byId.values()].sort((left, right) => (
    new Date(left.timestamp) - new Date(right.timestamp)
    || left.id - right.id
  )).slice(-MAX_DISPLAY_METRICS)
  return recent.sort((left, right) => (
    left.global_timestep - right.global_timestep
    || new Date(left.timestamp) - new Date(right.timestamp)
    || left.id - right.id
  ))
}

function LineChart({ title, points, color, emptyMessage }) {
  const width = 720
  const height = 220
  const padding = { top: 16, right: 18, bottom: 34, left: 52 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom
  const values = points.map((point) => point.value)
  const minValue = Math.min(0, ...values)
  const maxValue = Math.max(0, ...values)
  const valueRange = maxValue - minValue || 1
  const minTimestep = points[0]?.timestep ?? 0
  const maxTimestep = points[points.length - 1]?.timestep ?? 1
  const timestepRange = maxTimestep - minTimestep || 1
  const coordinates = points.map((point, index) => ({
    x: padding.left + ((point.timestep - minTimestep) / timestepRange) * chartWidth,
    y: padding.top + chartHeight - ((point.value - minValue) / valueRange) * chartHeight,
    key: `${point.timestep}-${index}`,
  }))

  return (
    <section className="metric-chart">
      <h4>{title}</h4>
      {points.length === 0 ? (
        <p className="chart-empty">{emptyMessage}</p>
      ) : (
        <>
          <svg
            className="chart-svg"
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label={`${title} versus training timestep`}
          >
            {[0, 0.5, 1].map((fraction) => {
              const y = padding.top + chartHeight * fraction
              const value = maxValue - valueRange * fraction
              return (
                <g key={fraction}>
                  <line
                    x1={padding.left}
                    x2={width - padding.right}
                    y1={y}
                    y2={y}
                    className="chart-gridline"
                  />
                  <text x={padding.left - 8} y={y + 4} textAnchor="end" className="chart-label">
                    {value.toFixed(1)}
                  </text>
                </g>
              )
            })}
            <polyline
              fill="none"
              stroke={color}
              strokeWidth="3"
              strokeLinejoin="round"
              strokeLinecap="round"
              points={coordinates.map(({ x, y }) => `${x},${y}`).join(' ')}
            />
            {coordinates.length === 1 && (
              <circle cx={coordinates[0].x} cy={coordinates[0].y} r="5" fill={color} />
            )}
            <text x={padding.left} y={height - 8} className="chart-label">
              {minTimestep.toLocaleString()} steps
            </text>
            <text x={width - padding.right} y={height - 8} textAnchor="end" className="chart-label">
              {maxTimestep.toLocaleString()} steps
            </text>
          </svg>
        </>
      )}
    </section>
  )
}

export default function ExperimentMonitor({ experiment }) {
  const [metrics, setMetrics] = useState([])
  const [liveStatus, setLiveStatus] = useState(null)
  const [connection, setConnection] = useState(() => (
    TERMINAL_STATUSES.has(experiment.status) ? 'disconnected' : 'connecting'
  ))
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState('')
  const [reconnectVersion, setReconnectVersion] = useState(0)

  useEffect(() => {
    let disposed = false
    let socket
    let reconnectTimer
    let stableTimer
    let retryCount = 0
    let terminalStatus = TERMINAL_STATUSES.has(experiment.status)

    async function refreshHistory() {
      try {
        const history = await getExperimentMetrics(experiment.id, { limit: 10000 })
        if (!disposed) {
          setMetrics((current) => mergeMetrics(current, history))
          setHistoryError('')
        }
      } catch (error) {
        if (!disposed) setHistoryError(error.message)
      } finally {
        if (!disposed) setHistoryLoading(false)
      }
    }

    function connect() {
      if (disposed || terminalStatus) return
      setConnection(retryCount === 0 ? 'connecting' : 'reconnecting')
      const currentSocket = new WebSocket(experimentMetricsWebSocketUrl(experiment.id))
      socket = currentSocket

      currentSocket.onopen = () => {
        if (disposed || socket !== currentSocket) return
        setConnection('connected')
        stableTimer = window.setTimeout(() => {
          retryCount = 0
        }, 15000)
        refreshHistory()
      }

      currentSocket.onmessage = (message) => {
        if (disposed || socket !== currentSocket) return
        let event
        try {
          event = JSON.parse(message.data)
        } catch (error) {
          console.error('Unable to parse a live metric message.', error)
          return
        }
        if (event.type === 'status') {
          setLiveStatus({ baseStatus: experiment.status, status: event.status })
          if (event.latest_metric) {
            setMetrics((current) => mergeMetrics(current, [event.latest_metric]))
          }
          if (TERMINAL_STATUSES.has(event.status)) {
            terminalStatus = true
            setConnection('disconnected')
            currentSocket.close()
          }
        } else if (event.type === 'metric' && event.metric) {
          setMetrics((current) => mergeMetrics(current, [event.metric]))
          setLiveStatus({
            baseStatus: experiment.status,
            status: event.metric.training_status,
          })
          if (TERMINAL_STATUSES.has(event.metric.training_status)) {
            terminalStatus = true
            setConnection('disconnected')
            currentSocket.close()
          }
        } else if (event.type === 'resync_required') {
          refreshHistory()
        }
      }

      currentSocket.onclose = () => {
        if (disposed || terminalStatus || socket !== currentSocket) return
        window.clearTimeout(stableTimer)
        refreshHistory()
        if (retryCount >= MAX_RECONNECT_ATTEMPTS) {
          setConnection('disconnected')
          return
        }
        setConnection('reconnecting')
        const delay = Math.min(30000, 1000 * (2 ** retryCount))
        retryCount += 1
        reconnectTimer = window.setTimeout(connect, delay)
      }

      currentSocket.onerror = () => currentSocket.close()
    }

    refreshHistory()
    connect()

    return () => {
      disposed = true
      window.clearTimeout(reconnectTimer)
      window.clearTimeout(stableTimer)
      socket?.close()
    }
  }, [experiment.id, experiment.status, reconnectVersion])

  const currentRunStart = useMemo(() => (
    [...metrics]
      .filter((metric) => metric.global_timestep === 0
        && metric.completed_episode_count === 0
        && metric.episode_reward === null)
      .sort((left, right) => (
        new Date(right.timestamp) - new Date(left.timestamp)
        || right.id - left.id
      ))[0] ?? null
  ), [metrics])
  const currentRunMetrics = currentRunStart
    ? metrics.filter((metric) => (
      new Date(metric.timestamp) >= new Date(currentRunStart.timestamp)
    ))
    : metrics
  const latestMetric = useMemo(() => (
    [...currentRunMetrics].sort((left, right) => (
      new Date(right.timestamp) - new Date(left.timestamp)
      || right.id - left.id
    ))[0] ?? null
  ), [currentRunMetrics])
  const latestEpisode = useMemo(() => (
    [...currentRunMetrics]
      .filter((metric) => metric.episode_reward !== null && metric.episode_reward !== undefined)
      .sort((left, right) => (
        new Date(right.timestamp) - new Date(left.timestamp)
        || right.id - left.id
      ))[0] ?? null
  ), [currentRunMetrics])
  const latestMean = useMemo(() => (
    [...currentRunMetrics]
      .filter((metric) => metric.recent_mean_episode_reward !== null
        && metric.recent_mean_episode_reward !== undefined)
      .sort((left, right) => (
        new Date(right.timestamp) - new Date(left.timestamp)
        || right.id - left.id
      ))[0]?.recent_mean_episode_reward ?? null
  ), [currentRunMetrics])

  const currentTimestep = latestMetric?.global_timestep ?? 0
  const status = liveStatus?.baseStatus === experiment.status
    ? liveStatus.status
    : experiment.status
  const connectionStatus = TERMINAL_STATUSES.has(status) ? 'disconnected' : connection
  const progress = Math.min(
    100,
    Math.max(0, (currentTimestep / experiment.total_timesteps) * 100),
  )
  const rewardPoints = metrics
    .filter((metric) => metric.episode_reward !== null && metric.episode_reward !== undefined)
    .map((metric) => ({
      timestep: metric.global_timestep,
      value: metric.episode_reward,
    }))
  const lengthPoints = metrics
    .filter((metric) => metric.episode_length !== null && metric.episode_length !== undefined)
    .map((metric) => ({
      timestep: metric.global_timestep,
      value: metric.episode_length,
    }))

  return (
    <section className="experiment-monitor" aria-label={`Monitoring for ${experiment.name}`}>
      <div className="monitor-heading">
        <div>
          <h4>Live monitoring</h4>
          <p>
            Live stream <span className={`connection-indicator ${connectionStatus}`} />
            {connectionStatus === 'connected' ? 'Connected' : connectionStatus === 'reconnecting'
              ? 'Reconnecting...' : connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
          </p>
          {connectionStatus === 'disconnected' && !TERMINAL_STATUSES.has(status) && (
            <button
              type="button"
              className="monitor-reconnect"
              onClick={() => setReconnectVersion((version) => version + 1)}
            >
              Retry live monitoring
            </button>
          )}
        </div>
        <span className={`monitor-status status-${status}`}>{status}</span>
      </div>

      {historyError && (
        <p className="monitor-error" role="status">
          Historical metrics could not be loaded: {historyError}
        </p>
      )}

      <div className="monitor-progress">
        <div className="progress-labels">
          <span>Progress</span>
          <strong>{currentTimestep.toLocaleString()} / {experiment.total_timesteps.toLocaleString()} steps</strong>
        </div>
        <div
          className="progress-track"
          role="progressbar"
          aria-valuenow={Math.round(progress)}
          aria-valuemin="0"
          aria-valuemax="100"
          aria-label="Training progress"
        >
          <span style={{ width: `${progress}%` }} />
        </div>
        <span className="progress-percent">{progress.toFixed(1)}%</span>
      </div>

      <div className="metric-summary">
        <div><span>Completed episodes</span><strong>{latestMetric?.completed_episode_count ?? 0}</strong></div>
        <div><span>Latest episode reward</span><strong>{latestEpisode ? latestEpisode.episode_reward.toFixed(2) : '--'}</strong></div>
        <div><span>Recent mean reward</span><strong>{latestMean === null ? '--' : latestMean.toFixed(2)}</strong></div>
        <div>
          <span>Training speed</span>
          <strong>
            {!latestMetric || latestMetric.timesteps_per_second === null
              ? '--'
              : `${latestMetric.timesteps_per_second.toFixed(1)} steps/s`}
          </strong>
        </div>
      </div>

      {historyLoading && metrics.length === 0 ? (
        <p className="monitor-empty">Loading saved training metrics...</p>
      ) : (
        <div className="metric-charts">
          <LineChart
            title="Episode reward vs timestep"
            points={rewardPoints}
            color="#71a7ff"
            emptyMessage="No completed episodes yet. Reward data will appear when an episode finishes."
          />
          <LineChart
            title="Episode length vs timestep"
            points={lengthPoints}
            color="#56d6ae"
            emptyMessage="Episode length will appear when an episode finishes."
          />
        </div>
      )}
    </section>
  )
}
