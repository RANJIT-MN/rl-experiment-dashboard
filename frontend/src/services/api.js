const API_BASE_URL = 'http://127.0.0.1:8000'

async function request(url, options, action) {
  let response

  try {
    response = await fetch(url, options)
  } catch {
    throw new Error(`Unable to ${action}: the backend at ${API_BASE_URL} could not be reached.`)
  }

  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string'
        ? body.detail
        : Array.isArray(body.detail)
          ? body.detail.map((issue) => issue.msg).join('; ')
          : ''
    } catch {
      // Fall back to the HTTP status when there is no JSON error body.
    }

    throw new Error(detail || `Unable to ${action}: server returned HTTP ${response.status}.`)
  }

  return response.json()
}

export async function checkBackendHealth() {
  return request(`${API_BASE_URL}/health`, undefined, 'check backend health')
}

export async function getExperiments() {
  return request(`${API_BASE_URL}/experiments`, undefined, 'load experiments')
}

export async function getExperimentMetrics(experimentId, options = {}) {
  const params = new URLSearchParams()
  if (options.limit !== undefined) params.set('limit', options.limit)
  if (options.fromTimestep !== undefined) params.set('from_timestep', options.fromTimestep)
  if (options.toTimestep !== undefined) params.set('to_timestep', options.toTimestep)
  const query = params.size ? `?${params.toString()}` : ''
  return request(
    `${API_BASE_URL}/experiments/${experimentId}/metrics${query}`,
    undefined,
    'load experiment metrics',
  )
}

export async function getRunMetrics(experimentId, runId, limit = 10000) {
  return request(
    `${API_BASE_URL}/experiments/${experimentId}/runs/${runId}/metrics?limit=${limit}`,
    undefined,
    'load training run metrics',
  )
}

export async function evaluateRun(runId, evaluation) {
  return request(`${API_BASE_URL}/experiments/runs/${runId}/evaluate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(evaluation),
  }, 'start policy evaluation')
}

export async function compareExperiments(experimentIds) {
  return request(`${API_BASE_URL}/experiments/compare`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ experiment_ids: experimentIds }),
  }, 'compare experiments')
}

export async function exportComparisonMetrics(experimentIds) {
  const url = `${API_BASE_URL}/experiments/compare/export`
  let response
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ experiment_ids: experimentIds }),
    })
  } catch {
    throw new Error('Unable to export metrics: the backend at http://127.0.0.1:8000 could not be reached.')
  }

  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string'
        ? body.detail
        : Array.isArray(body.detail)
          ? body.detail.map((issue) => issue.msg).join('; ')
          : ''
    } catch {
      // Fall back to the HTTP status when there is no JSON error body.
    }
    throw new Error(detail || `Unable to export metrics: server returned HTTP ${response.status}.`)
  }

  const blob = await response.blob()
  const disposition = response.headers.get('Content-Disposition') ?? ''
  const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1]
    ?? 'experiment-comparison.csv'
  const downloadUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = downloadUrl
  link.download = filename
  document.body.append(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(downloadUrl)
}

export function experimentMetricsWebSocketUrl(experimentId) {
  return `${API_BASE_URL.replace(/^http/, 'ws')}/experiments/${experimentId}/metrics/ws`
}

export async function createExperiment(experiment) {
  return request(`${API_BASE_URL}/experiments`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(experiment),
  }, 'create experiment')
}

export async function startExperiment(experimentId) {
  return request(`${API_BASE_URL}/experiments/${experimentId}/start`, {
    method: 'POST',
  }, 'start training')
}

export async function stopExperiment(experimentId) {
  return request(`${API_BASE_URL}/experiments/${experimentId}/stop`, {
    method: 'POST',
  }, 'request training stop')
}