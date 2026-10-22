import { useEffect, useMemo, useState } from 'react'
import './App.css'
import ExperimentMonitor from './components/ExperimentMonitor'
import ExperimentComparison from './components/ExperimentComparison'
import RunMonitor from './components/RunMonitor'
import {
  checkBackendHealth,
  getExperiments,
  createExperiment,
  evaluateRun,
  startExperiment,
  stopExperiment,
  compareExperiments,
  exportComparisonMetrics,
} from './services/api'

const DEFAULT_FORM = {
  name: '',
  environment: 'InvertedPendulum-v5',
  algorithm: 'PPO',
  total_timesteps: '1000',
  learning_rate: '0.0003',
  n_steps: '2048',
  batch_size: '64',
  gamma: '0.99',
  n_epochs: '10',
  base_seed: '42',
  run_count: '3',
}

const PPO_CONFIGURATION_FIELDS = [
  ['learning_rate', 'Learning rate'],
  ['n_steps', 'Rollout length'],
  ['batch_size', 'Batch size'],
  ['gamma', 'Discount factor'],
  ['n_epochs', 'PPO epochs'],
  ['base_seed', 'Base random seed'],
  ['run_count', 'Runs per start'],
]

function App() {
  const [backendStatus, setBackendStatus] = useState('Checking...')
  const [experiments, setExperiments] = useState([])
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const [form, setForm] = useState(DEFAULT_FORM)

  const [creating, setCreating] = useState(false)
  const [activeActionId, setActiveActionId] = useState(null)
  const [selectedExperimentIds, setSelectedExperimentIds] = useState([])
  const [comparison, setComparison] = useState(null)
  const [comparisonError, setComparisonError] = useState('')
  const [comparisonLoading, setComparisonLoading] = useState(false)
  const [exportLoading, setExportLoading] = useState(false)
  const selectedExperiments = useMemo(() => {
    const selectedIds = new Set(selectedExperimentIds)
    return experiments.filter((experiment) => selectedIds.has(experiment.id))
  }, [experiments, selectedExperimentIds])

  useEffect(() => {
    loadDashboard()
  }, [])

  const hasActiveWork = experiments.some((experiment) =>
    ['pending', 'running', 'stopping'].includes(experiment.status)
      || (experiment.runs ?? []).some((run) => (
        ['pending', 'running'].includes(run.evaluation?.status)
      )),
  )

  useEffect(() => {
    if (!hasActiveWork) return undefined

    const intervalId = window.setInterval(async () => {
      try {
        setExperiments(await getExperiments())
      } catch (err) {
        console.error(err)
        setError(err.message)
      }
    }, 1500)

    return () => window.clearInterval(intervalId)
  }, [hasActiveWork])

  async function loadDashboard() {
    try {
      await checkBackendHealth()
      setBackendStatus('Connected')

      const data = await getExperiments()
      setExperiments(data)
      setError('')
    } catch (err) {
      console.error(err)
      if (err.message.includes('could not be reached')) {
        setBackendStatus('Offline')
      }
      setError(err.message)
    }
  }

  function handleChange(event) {
    const { name, value } = event.target

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }))
  }

  async function handleCreateExperiment(event) {
    event.preventDefault()

    setError('')
    setSuccess('')

    const numericValues = Object.fromEntries(
      ['total_timesteps', 'learning_rate', 'n_steps', 'batch_size', 'gamma', 'n_epochs', 'base_seed', 'run_count']
        .map((field) => [field, Number(form[field])]),
    )
    const integerFields = ['total_timesteps', 'n_steps', 'batch_size', 'n_epochs', 'base_seed', 'run_count']
    if (!form.name.trim()) {
      setError('Enter an experiment name.')
      return
    }
    if (Object.values(numericValues).some((value) => !Number.isFinite(value))) {
      setError('Enter a valid number in every numeric field.')
      return
    }
    if (integerFields.some((field) => !Number.isInteger(numericValues[field]))) {
      setError('Total timesteps, rollout length, batch size, and PPO epochs must be whole numbers.')
      return
    }
    if (numericValues.total_timesteps < 1) {
      setError('Total timesteps must be a positive whole number.')
      return
    }
    if (numericValues.learning_rate <= 0) {
      setError('Learning rate must be greater than zero.')
      return
    }
    if (numericValues.n_steps <= 1) {
      setError('Rollout length must be at least 2 for PPO advantage normalization.')
      return
    }
    if (numericValues.batch_size <= 1) {
      setError('Batch size must be at least 2 for PPO advantage normalization.')
      return
    }
    if (numericValues.n_steps % numericValues.batch_size !== 0) {
      setError('Batch size must divide the rollout length evenly (for example, 64 into 2048).')
      return
    }
    if (numericValues.gamma <= 0 || numericValues.gamma > 1) {
      setError('Discount factor must be greater than zero and no greater than 1.')
      return
    }
    if (numericValues.n_epochs < 1) {
      setError('PPO epochs must be a positive whole number.')
      return
    }
    if (numericValues.base_seed < 0 || numericValues.base_seed > 4294967295) {
      setError('Base seed must be a whole number from 0 to 4,294,967,295.')
      return
    }
    if (numericValues.run_count < 1 || numericValues.run_count > 20) {
      setError('Choose between 1 and 20 repeated runs.')
      return
    }

    setCreating(true)
    try {
      const experiment = await createExperiment({
        name: form.name.trim(),
        environment: form.environment,
        algorithm: form.algorithm,
        ...numericValues,
      })

      setExperiments((previous) => [...previous, experiment])

      setSuccess(`Experiment "${experiment.name}" created successfully.`)

      setForm(DEFAULT_FORM)
    } catch (err) {
      console.error(err)
      setError(err.message)
    } finally {
      setCreating(false)
    }
  }

  async function handleTrainingAction(experiment, action) {
    setActiveActionId(experiment.id)
    setError('')
    setSuccess('')

    try {
      const result = action === 'start'
        ? await startExperiment(experiment.id)
        : await stopExperiment(experiment.id)
      const data = await getExperiments()
      setExperiments(data)
      setSuccess(result.message)
    } catch (err) {
      console.error(err)
      setError(err.message)
      try {
        setExperiments(await getExperiments())
      } catch (refreshError) {
        console.error(refreshError)
      }
    } finally {
      setActiveActionId(null)
    }
  }

  async function handleEvaluateRun(run) {
    setActiveActionId(`evaluation-${run.id}`)
    setError('')
    setSuccess('')
    try {
      await evaluateRun(run.id, { episode_count: 10, base_seed: 100000 })
      setExperiments(await getExperiments())
      setSuccess(`Policy evaluation started for run ${run.run_index + 1}.`)
    } catch (err) {
      console.error(err)
      setError(err.message)
    } finally {
      setActiveActionId(null)
    }
  }

  function handleComparisonSelection(event) {
    const experimentId = Number(event.target.value)
    const selected = event.target.checked
    setComparison(null)
    setComparisonError('')
    if (!selected) {
      setSelectedExperimentIds((current) => current.filter((id) => id !== experimentId))
      return
    }
    if (selectedExperimentIds.includes(experimentId)) return
    if (selectedExperimentIds.length >= 10) {
      setComparisonError('Select no more than 10 experiments at a time.')
      return
    }
    setSelectedExperimentIds((current) => [...current, experimentId])
  }

  async function handleCompareExperiments() {
    setComparisonLoading(true)
    setComparisonError('')
    try {
      const result = await compareExperiments(selectedExperiments.map(({ id }) => id))
      setComparison(result)
    } catch (err) {
      console.error(err)
      setComparisonError(err.message)
      setComparison(null)
    } finally {
      setComparisonLoading(false)
    }
  }

  async function handleExportComparison() {
    setExportLoading(true)
    setComparisonError('')
    try {
      await exportComparisonMetrics(selectedExperiments.map(({ id }) => id))
    } catch (err) {
      console.error(err)
      setComparisonError(err.message)
    } finally {
      setExportLoading(false)
    }
  }

  return (
    <div className="dashboard">

      {/* HEADER */}
      <header className="header">
        <div>
          <h1>RL Experiment Dashboard</h1>
          <p>Real-Time Reinforcement Learning Experiment Tracker</p>
        </div>

        <div className="system-status">
          <span
            className={
              backendStatus === 'Connected'
                ? 'status-dot online'
                : 'status-dot offline'
            }
          ></span>

          System {backendStatus}
        </div>
      </header>

      <main className="content">

        {/* HERO */}
        <section className="hero-card">
          <h2>MuJoCo + PPO Foundation</h2>

          <p>
            Reinforcement learning simulation foundation successfully
            initialized.
          </p>
        </section>

        {/* SYSTEM STATUS */}
        <section className="status-grid">

          <div className="status-card">
            <h3>Frontend</h3>
            <strong>Running</strong>
            <p>React + Vite</p>
          </div>

          <div className="status-card">
            <h3>Backend</h3>
            <strong>{backendStatus}</strong>
            <p>FastAPI</p>
          </div>

          <div className="status-card">
            <h3>Simulation</h3>
            <strong>Validated</strong>
            <p>MuJoCo</p>
          </div>

          <div className="status-card">
            <h3>Environment</h3>
            <strong>Validated</strong>
            <p>InvertedPendulum-v5</p>
          </div>

          <div className="status-card">
            <h3>Algorithm</h3>
            <strong>Validated</strong>
            <p>PPO</p>
          </div>

        </section>

        {/* CREATE EXPERIMENT */}
        <section className="create-section">

          <div className="section-title">
            <div>
              <h2>Create Experiment</h2>
              <p>Start a new reinforcement learning experiment</p>
            </div>
          </div>

          <form
            className="experiment-form"
            onSubmit={handleCreateExperiment}
            noValidate
          >

            <div className="form-group">
              <label htmlFor="name">
                Experiment Name
              </label>

              <input
                id="name"
                name="name"
                type="text"
                placeholder="Example: PPO Training Run 01"
                value={form.name}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="environment">
                Environment
              </label>

              <select
                id="environment"
                name="environment"
                value={form.environment}
                onChange={handleChange}
              >
                <option value="InvertedPendulum-v5">
                  InvertedPendulum-v5
                </option>

              </select>
            </div>

            <div className="form-group">
              <label htmlFor="algorithm">
                Algorithm
              </label>

              <select
                id="algorithm"
                name="algorithm"
                value={form.algorithm}
                onChange={handleChange}
              >
                <option value="PPO">PPO</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="total_timesteps">
                Total Timesteps
              </label>

              <input
                id="total_timesteps"
                name="total_timesteps"
                type="number"
                min="1"
                value={form.total_timesteps}
                onChange={handleChange}
                step="1"
                required
              />
              <small>Number of environment steps requested for training.</small>
            </div>

            <div className="form-group">
              <label htmlFor="learning_rate">Learning rate</label>
              <input
                id="learning_rate"
                name="learning_rate"
                type="number"
                min="0"
                step="any"
                value={form.learning_rate}
                onChange={handleChange}
                required
              />
              <small>Optimizer step size; must be greater than 0.</small>
            </div>

            <div className="form-group">
              <label htmlFor="n_steps">Rollout length (n_steps)</label>
              <input
                id="n_steps"
                name="n_steps"
                type="number"
                min="2"
                step="1"
                value={form.n_steps}
                onChange={handleChange}
                required
              />
              <small>Steps collected per PPO update; must divide evenly by batch size.</small>
            </div>

            <div className="form-group">
              <label htmlFor="batch_size">Batch size</label>
              <input
                id="batch_size"
                name="batch_size"
                type="number"
                min="2"
                step="1"
                value={form.batch_size}
                onChange={handleChange}
                required
              />
              <small>Minibatch size; must evenly divide the rollout buffer.</small>
            </div>

            <div className="form-group">
              <label htmlFor="gamma">Discount factor (gamma)</label>
              <input
                id="gamma"
                name="gamma"
                type="number"
                min="0"
                max="1"
                step="any"
                value={form.gamma}
                onChange={handleChange}
                required
              />
              <small>Future-reward discount; greater than 0 and at most 1.</small>
            </div>

            <div className="form-group">
              <label htmlFor="n_epochs">PPO epochs (n_epochs)</label>
              <input
                id="n_epochs"
                name="n_epochs"
                type="number"
                min="1"
                step="1"
                value={form.n_epochs}
                onChange={handleChange}
                required
              />
              <small>Optimization passes over each collected rollout.</small>
            </div>

            <div className="form-group">
              <label htmlFor="base_seed">Base random seed</label>
              <input
                id="base_seed"
                name="base_seed"
                type="number"
                min="0"
                max="4294967295"
                step="1"
                value={form.base_seed}
                onChange={handleChange}
                required
              />
              <small>Run seeds increment from this value; identical seeds do not guarantee bitwise determinism.</small>
            </div>

            <div className="form-group">
              <label htmlFor="run_count">Repeated runs</label>
              <input
                id="run_count"
                name="run_count"
                type="number"
                min="1"
                max="20"
                step="1"
                value={form.run_count}
                onChange={handleChange}
                required
              />
              <small>Runs execute sequentially, one PPO training job at a time.</small>
            </div>

            <button
              className="create-button"
              type="submit"
              disabled={creating}
            >
              {creating ? 'Creating...' : 'Create Experiment'}
            </button>

          </form>

          {success && (
            <div className="success-box">
              ✓ {success}
            </div>
          )}

          {error && (
            <div className="error-box">
              ✕ {error}
            </div>
          )}

        </section>

        {/* EXPERIMENT LIST */}
        <section className="experiments-section">

          <div className="section-header">

            <div>
              <h2>Experiments</h2>
              <p>
                Experiments stored in the RL dashboard database
              </p>
            </div>

            <span className="experiment-count">
              {experiments.length} Experiment
              {experiments.length !== 1 ? 's' : ''}
            </span>

          </div>

          {experiments.length === 0 ? (

            <div className="empty-state">
              No experiments found.
            </div>

          ) : (

            <div className="experiment-list">

              {experiments.map((experiment) => (

                <div
                  className="experiment-card"
                  key={experiment.id}
                >

                  <div className="experiment-main">

                    <h3>{experiment.name}</h3>

                    <div className="experiment-details">

                      <div>
                        <span>Environment</span>

                    {experiment.failure_message && (
                      <p className="experiment-failure" role="status">
                        {experiment.failure_message}
                      </p>
                    )}
                        <strong>
                          {experiment.environment}
                        </strong>
                      </div>

                      <div>
                        <span>Algorithm</span>
                        <strong>
                          {experiment.algorithm}
                        </strong>
                      </div>

                      <div>
                        <span>Status</span>
                        <strong className="created">
                          {experiment.status}
                        </strong>
                      </div>

                      <div>
                        <span>Timesteps</span>
                        <strong>
                          {experiment.total_timesteps}
                        </strong>
                      </div>

                    </div>

                    <div className="experiment-configuration" aria-label="Saved PPO configuration and run settings">
                      <span className="configuration-title">Saved PPO configuration and run settings</span>
                      {PPO_CONFIGURATION_FIELDS.map(([field, label]) => (
                        <div key={field}>
                          <span>{label}</span>
                          <strong>
                            {experiment[field] === null || experiment[field] === undefined
                              ? 'Not recorded (legacy)'
                              : experiment[field]}
                          </strong>
                        </div>
                      ))}
                    </div>

                  </div>

                  <div className="experiment-id">
                    #{experiment.id}
                  </div>

                  <div className="training-controls">
                    <button
                      type="button"
                      onClick={() => handleTrainingAction(experiment, 'start')}
                      disabled={
                        activeActionId === experiment.id ||
                        !['created', 'failed', 'stopped'].includes(experiment.status)
                      }
                    >
                      {activeActionId === experiment.id ? 'Working...' : 'Start Training'}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleTrainingAction(experiment, 'stop')}
                      disabled={
                        activeActionId === experiment.id ||
                        !['pending', 'running'].includes(experiment.status)
                      }
                    >
                      Stop Training
                    </button>
                    {experiment.status === 'stopping' && (
                      <span role="status">Waiting for training to stop...</span>
                    )}
                  </div>

                  {experiment.runs?.length > 0 && (
                    <section className="training-runs" aria-label={`Training runs for ${experiment.name}`}>
                      <h4>Repeated training runs</h4>
                      <p>Run seeds are derived as (base seed + run index) modulo 2^32.</p>
                      {experiment.runs.map((run) => (
                        <article className="training-run-card" key={run.id}>
                          <div className="training-run-heading">
                            <strong>Run {run.run_index + 1} · seed {run.effective_seed}</strong>
                            <span className={`monitor-status status-${run.status}`}>{run.status}</span>
                          </div>
                          <p>
                            {run.actual_timesteps === null || run.actual_timesteps === undefined
                              ? `Requested ${run.requested_timesteps.toLocaleString()} timesteps`
                              : `${run.actual_timesteps.toLocaleString()} / ${run.requested_timesteps.toLocaleString()} actual timesteps`}
                          </p>
                          {run.failure_message && <p className="experiment-failure">{run.failure_message}</p>}
                          {run.evaluation && (
                            <p className="evaluation-result" role="status">
                              Evaluation {run.evaluation.status}
                              {run.evaluation.status === 'completed'
                                ? ` · mean reward ${run.evaluation.mean_episode_reward.toFixed(2)}, mean length ${run.evaluation.mean_episode_length.toFixed(1)} across ${run.evaluation.episode_count} episodes`
                                : run.evaluation.failure_message ? ` · ${run.evaluation.failure_message}` : ''}
                            </p>
                          )}
                          {run.status === 'completed' && !run.evaluation && (
                            <button
                              type="button"
                              className="evaluation-button"
                              onClick={() => handleEvaluateRun(run)}
                              disabled={activeActionId === `evaluation-${run.id}`}
                            >
                              {activeActionId === `evaluation-${run.id}` ? 'Starting evaluation...' : 'Evaluate saved policy'}
                            </button>
                          )}
                          <RunMonitor experiment={experiment} run={run} />
                        </article>
                      ))}
                    </section>
                  )}

                  <ExperimentMonitor experiment={experiment} />
                </div>

              ))}

            </div>

          )}

        </section>

        <section className="comparison-section" aria-labelledby="comparison-section-title">
          <div className="section-header">
            <div>
              <h2 id="comparison-section-title">Experiment comparison</h2>
              <p>Select two or more experiments to compare their saved configuration and recorded performance.</p>
            </div>
            <span className="experiment-count">{selectedExperimentIds.length} selected</span>
          </div>

          {experiments.length === 0 ? (
            <div className="empty-state">Create experiments before comparing runs.</div>
          ) : (
            <>
              <div className="comparison-selection-list">
                {experiments.map((experiment) => (
                  <label className="comparison-selection-item" key={experiment.id}>
                    <input
                      type="checkbox"
                      value={experiment.id}
                      checked={selectedExperimentIds.includes(experiment.id)}
                      onChange={handleComparisonSelection}
                    />
                    <span>
                      <strong>{experiment.name}</strong>
                      <small>#{experiment.id} · {experiment.status}</small>
                    </span>
                  </label>
                ))}
              </div>

              <div className="comparison-actions">
                <p>
                  {selectedExperimentIds.length < 2
                    ? 'Select at least two experiments to compare.'
                    : `${selectedExperimentIds.length} selected; up to 10 can be compared at once.`}
                </p>
                <button
                  type="button"
                  className="create-button comparison-action-button"
                  onClick={handleCompareExperiments}
                  disabled={selectedExperimentIds.length < 2 || comparisonLoading}
                >
                  {comparisonLoading ? 'Comparing...' : 'Compare selected'}
                </button>
                <button
                  type="button"
                  className="export-button"
                  onClick={handleExportComparison}
                  disabled={selectedExperimentIds.length < 2 || exportLoading}
                >
                  {exportLoading ? 'Preparing CSV...' : 'Export selected metrics CSV'}
                </button>
              </div>

              {comparisonError && (
                <div className="error-box" role="alert">{comparisonError}</div>
              )}
              {comparison && (
                <ExperimentComparison comparison={comparison} />
              )}
            </>
          )}
        </section>

      </main>

    </div>
  )
}

export default App