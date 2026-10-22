import { useMemo } from 'react'

const SERIES_STYLES = [
  { color: '#71a7ff', shape: 'circle' },
  { color: '#56d6ae', shape: 'square' },
  { color: '#ffb86b', shape: 'diamond' },
  { color: '#e88aff', shape: 'triangle' },
  { color: '#ff7785', shape: 'cross' },
  { color: '#d5dc72', shape: 'circle' },
  { color: '#74d6e8', shape: 'square' },
  { color: '#ffa6d2', shape: 'diamond' },
  { color: '#c1a8ff', shape: 'triangle' },
  { color: '#b7d889', shape: 'cross' },
]

const HYPERPARAMETERS = [
  ['learning_rate', 'Learning rate'],
  ['n_steps', 'n_steps'],
  ['batch_size', 'batch_size'],
  ['gamma', 'gamma'],
  ['n_epochs', 'n_epochs'],
  ['base_seed', 'Base seed'],
  ['run_count', 'Runs per start'],
]

function formatValue(value, digits = 3) {
  if (value === null || value === undefined) return 'Not recorded'
  return digits === null ? String(value) : value.toFixed(digits)
}

function marker(shape, x, y, color, size = 5) {
  if (shape === 'square') {
    return <rect x={x - size} y={y - size} width={size * 2} height={size * 2} fill={color} />
  }
  if (shape === 'diamond') {
    return <path d={`M ${x} ${y - size - 1} L ${x + size + 1} ${y} L ${x} ${y + size + 1} L ${x - size - 1} ${y} Z`} fill={color} />
  }
  if (shape === 'triangle') {
    return <path d={`M ${x} ${y - size - 1} L ${x + size + 1} ${y + size} L ${x - size - 1} ${y + size} Z`} fill={color} />
  }
  if (shape === 'cross') {
    return (
      <path
        d={`M ${x - size} ${y - size} L ${x + size} ${y + size} M ${x + size} ${y - size} L ${x - size} ${y + size}`}
        fill="none"
        stroke={color}
        strokeWidth="2"
      />
    )
  }
  return <circle cx={x} cy={y} r={size} fill={color} />
}

function ScatterComparisonChart({ title, valueLabel, valueField, experiments }) {
  const width = 760
  const height = 300
  const padding = { top: 26, right: 22, bottom: 52, left: 66 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  const series = experiments.map((item, index) => ({
    item,
    style: SERIES_STYLES[index % SERIES_STYLES.length],
    points: item.curve_metrics
      .filter((metric) => metric[valueField] !== null && metric[valueField] !== undefined)
      .map((metric) => ({
        timestep: metric.global_timestep,
        value: metric[valueField],
        timestamp: metric.timestamp,
      })),
  }))
  const allPoints = series.flatMap((entry) => entry.points)
  const minTimestep = Math.min(0, ...allPoints.map((point) => point.timestep))
  const maxTimestep = Math.max(1, ...allPoints.map((point) => point.timestep))
  const minValue = Math.min(0, ...allPoints.map((point) => point.value))
  const maxValue = Math.max(1, ...allPoints.map((point) => point.value))
  const timestepRange = maxTimestep - minTimestep || 1
  const valueRange = maxValue - minValue || 1

  return (
    <section className="comparison-chart-card">
      <h3>{title}</h3>
      {allPoints.length === 0 ? (
        <p className="comparison-empty">No persisted {valueLabel.toLowerCase()} observations are available for these experiments.</p>
      ) : (
        <>
          <div className="comparison-chart-scroll">
            <svg
              className="comparison-chart"
              viewBox={`0 0 ${width} ${height}`}
              role="img"
              aria-label={`${title}. Discrete points show observed episode metrics only; points are not connected or interpolated.`}
            >
              <text
                x="18"
                y={height / 2}
                transform={`rotate(-90 18 ${height / 2})`}
                className="comparison-axis-label"
                textAnchor="middle"
              >
                {valueLabel}
              </text>
              {[0, 0.5, 1].map((fraction) => {
                const y = padding.top + plotHeight * (1 - fraction)
                const value = minValue + valueRange * fraction
                return (
                  <g key={fraction}>
                    <line
                      x1={padding.left}
                      x2={width - padding.right}
                      y1={y}
                      y2={y}
                      className="chart-gridline"
                    />
                    <text x={padding.left - 9} y={y + 4} textAnchor="end" className="chart-label">
                      {value.toFixed(1)}
                    </text>
                  </g>
                )
              })}
              {series.map(({ item, style, points }) => points.map((point, pointIndex) => {
                const x = padding.left + ((point.timestep - minTimestep) / timestepRange) * plotWidth
                const y = padding.top + plotHeight - ((point.value - minValue) / valueRange) * plotHeight
                return (
                  <g
                    key={`${item.experiment.id}-${pointIndex}`}
                    role="img"
                    tabIndex="0"
                    aria-label={`${item.experiment.name}, timestep ${point.timestep}, ${valueLabel.toLowerCase()} ${point.value}`}
                  >
                    <title>
                      {item.experiment.name} (#{item.experiment.id}) — timestep {point.timestep.toLocaleString()}, {valueLabel.toLowerCase()} {point.value}, {new Date(point.timestamp).toLocaleString()}
                    </title>
                    {marker(style.shape, x, y, style.color)}
                  </g>
                )
              }))}
              <text x={padding.left} y={height - 12} className="chart-label">
                {minTimestep.toLocaleString()}
              </text>
              <text x={width - padding.right} y={height - 12} textAnchor="end" className="chart-label">
                {maxTimestep.toLocaleString()}
              </text>
              <text
                x={padding.left + plotWidth / 2}
                y={height - 12}
                textAnchor="middle"
                className="comparison-axis-label"
              >
                Global timestep
              </text>
            </svg>
          </div>
          <ul className="comparison-legend" aria-label={`${title} legend`}>
            {series.map(({ item, style, points }) => (
              <li key={item.experiment.id}>
                <svg className="comparison-legend-marker" viewBox="0 0 16 16" aria-hidden="true">
                  {marker(style.shape, 8, 8, style.color, 4)}
                </svg>
                <span>{item.experiment.name} (#{item.experiment.id}) — {points.length} observed points</span>
              </li>
            ))}
          </ul>
          {series.some(({ item }) => item.curve_truncated) && (
            <p className="comparison-note">Chart points are capped at 2,000 per experiment; summary statistics use all persisted completed episodes.</p>
          )}
          <p className="comparison-note">Points are discrete observations; no values are interpolated between episodes.</p>
        </>
      )}
    </section>
  )
}

function HyperparameterDifferences({ experiments }) {
  const differences = HYPERPARAMETERS.filter(([field]) => (
    new Set(experiments.map(({ experiment }) => (
      experiment[field] === null || experiment[field] === undefined
        ? 'not-recorded'
        : String(experiment[field])
    ))).size > 1
  ))

  if (differences.length === 0) {
    return <p className="comparison-note">No recorded hyperparameter differences among these experiments.</p>
  }

  return (
    <ul className="comparison-differences">
      {differences.map(([field, label]) => (
        <li key={field}>
          <strong>{label}:</strong>{' '}
          {experiments.map(({ experiment }) => (
            `${experiment.name} (#${experiment.id}): ${
              experiment[field] === null || experiment[field] === undefined
                ? 'Not recorded'
                : experiment[field]
            }`
          )).join(' · ')}
        </li>
      ))}
    </ul>
  )
}

export default function ExperimentComparison({ comparison }) {
  const experiments = comparison.experiments
  const completed = experiments.filter(({ experiment }) => experiment.status === 'completed')
  const incomplete = experiments.filter(({ experiment }) => experiment.status !== 'completed')
  const ranked = useMemo(() => (
    experiments
      .filter((item) => item.mean_episode_reward !== null)
      .sort((left, right) => right.mean_episode_reward - left.mean_episode_reward)
  ), [experiments])

  return (
    <section className="comparison-results" aria-label="Experiment comparison results">
      <div className="comparison-results-header">
        <div>
          <h3>Comparison results</h3>
          <p>{comparison.metric_definition}</p>
        </div>
      </div>

      <div className="comparison-summary">
        <div>
          <h4>Completed successfully</h4>
          <p>{completed.length ? completed.map(({ experiment }) => `${experiment.name} (#${experiment.id})`).join(', ') : 'No selected experiment has completed successfully.'}</p>
        </div>
        <div>
          <h4>Recorded mean rewards</h4>
          {experiments.map((item) => (
            <p key={item.experiment.id}>
              {item.experiment.name} (#{item.experiment.id}): {formatValue(item.mean_episode_reward, 2)}
              {item.mean_episode_reward !== null && ` across ${item.completed_episode_count} completed episode observations`}
            </p>
          ))}
          {ranked.length > 0 && (
            <p className="comparison-note">
              Highest training-reward mean in this selection: {ranked[0].experiment.name} (#{ranked[0].experiment.id}), based on {ranked[0].completed_episode_count} persisted training episode observations. This is descriptive only and does not establish general superiority.
            </p>
          )}
        </div>
        <div>
          <h4>Separate policy evaluation across repeated runs</h4>
          {experiments.map((item) => (
            <p key={item.experiment.id}>
              {item.experiment.name} (#{item.experiment.id}):{' '}
              {item.evaluated_run_count === 0
                ? 'No completed policy evaluations'
                : `${formatValue(item.evaluation_reward_mean, 2)} mean per-run evaluation reward across ${item.evaluated_run_count} evaluated runs`}
              {item.evaluated_run_count > 1
                ? ` (sample SD ${formatValue(item.evaluation_reward_sample_stddev, 2)})`
                : item.evaluated_run_count === 1 ? ' (sample SD unavailable with one run)' : ''}
              {`; ${formatValue(item.mean_evaluation_episode_length, 2)} mean episode length across ${item.evaluation_episode_count} actual evaluation episodes`}
            </p>
          ))}
          <p className="comparison-note">
            Reward mean and sample SD use completed per-run evaluation means (sample SD requires n ≥ 2). Episode length is pooled across the latest successful evaluation for each evaluated run. Training rewards are not substituted for evaluation results.
          </p>
        </div>
        <div>
          <h4>Latest recorded training speed</h4>
          {experiments.map((item) => (
            <p key={item.experiment.id}>
              {item.experiment.name} (#{item.experiment.id}): {item.latest_training_speed === null ? 'Not recorded' : `${item.latest_training_speed.toFixed(1)} steps/s`}
            </p>
          ))}
        </div>
        <div>
          <h4>Configuration differences</h4>
          <HyperparameterDifferences experiments={experiments} />
        </div>
        {incomplete.length > 0 && (
          <div>
            <h4>Incomplete or unsuccessful runs</h4>
            <p>
              {incomplete.map(({ experiment }) => `${experiment.name} (#${experiment.id}): ${experiment.status}`).join(', ')}
            </p>
          </div>
        )}
        {experiments.some((item) => item.completed_episode_count === 0) && (
          <div>
            <h4>Missing episode metrics</h4>
            <p>
              {experiments
                .filter((item) => item.completed_episode_count === 0)
                .map(({ experiment }) => `${experiment.name} (#${experiment.id}) has no persisted completed-episode rewards`)
                .join('; ')}
            </p>
          </div>
        )}
      </div>

      <div className="comparison-table-scroll">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Experiment</th>
              <th>Environment</th>
              <th>Algorithm</th>
              <th>Status</th>
              <th>Requested steps</th>
              {HYPERPARAMETERS.map(([field, label]) => <th key={field}>{label}</th>)}
              <th>Episodes</th>
              <th>Latest reward</th>
              <th>Mean reward</th>
              <th>Max reward</th>
              <th>Min reward</th>
              <th>Mean length</th>
              <th>Latest speed</th>
              <th>Training runs</th>
              <th>Completed runs</th>
              <th>Evaluated runs</th>
              <th>Mean evaluation reward</th>
              <th>Evaluation reward sample SD</th>
              <th>Evaluation episodes</th>
              <th>Mean evaluation length</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((item) => (
              <tr key={item.experiment.id}>
                <th scope="row">{item.experiment.name} (#{item.experiment.id})</th>
                <td>{item.experiment.environment}</td>
                <td>{item.experiment.algorithm}</td>
                <td>{item.experiment.status}</td>
                <td>{item.experiment.total_timesteps.toLocaleString()}</td>
                {HYPERPARAMETERS.map(([field]) => (
                  <td key={field}>{formatValue(item.experiment[field], null)}</td>
                ))}
                <td>{item.completed_episode_count}</td>
                <td>{formatValue(item.latest_episode_reward, 2)}</td>
                <td>{formatValue(item.mean_episode_reward, 2)}</td>
                <td>{formatValue(item.maximum_episode_reward, 2)}</td>
                <td>{formatValue(item.minimum_episode_reward, 2)}</td>
                <td>{formatValue(item.mean_episode_length, 2)}</td>
                <td>{item.latest_training_speed === null ? 'Not recorded' : `${item.latest_training_speed.toFixed(1)} steps/s`}</td>
                <td>{item.training_run_count}</td>
                <td>{item.completed_training_run_count}</td>
                <td>{item.evaluated_run_count}</td>
                <td>{formatValue(item.evaluation_reward_mean, 2)}</td>
                <td>
                  {item.evaluated_run_count > 1
                    ? formatValue(item.evaluation_reward_sample_stddev, 2)
                    : item.evaluated_run_count === 1 ? 'n=1; unavailable' : 'Not evaluated'}
                </td>
                <td>{item.evaluation_episode_count}</td>
                <td>{formatValue(item.mean_evaluation_episode_length, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="comparison-charts">
        <ScatterComparisonChart
          title="Episode rewards over training"
          valueLabel="Episode reward"
          valueField="episode_reward"
          experiments={experiments}
        />
        <ScatterComparisonChart
          title="Episode lengths over training"
          valueLabel="Episode length"
          valueField="episode_length"
          experiments={experiments}
        />
      </div>
    </section>
  )
}
