/**
 * src/api.js
 *
 * Thin wrapper around axios for all DSBF API endpoints.
 * Import and call these functions from Vue components - never
 * write fetch/axios calls directly in components.
 *
 * All functions return the response data directly (not the axios response
 * object), so callers just get the plain JSON they expect.
 */

import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

// ── Datasets ──────────────────────────────────────────────────────────────────

/** List all datasets with run summary info. */
export function listDatasets() {
  return client.get('/datasets').then(r => r.data)
}

/** Get a single dataset by name. */
export function getDataset(name) {
  return client.get(`/datasets/${name}`).then(r => r.data)
}

/** List all runs for a dataset, newest first. */
export function listRuns(datasetName) {
  return client.get(`/datasets/${datasetName}/runs`).then(r => r.data)
}

// ── Runs ──────────────────────────────────────────────────────────────────────

/** Get run metadata by run_key (no task results). */
export function getRun(runKey) {
  return client.get(`/runs/${runKey}`).then(r => r.data)
}

/** Get all task results for a run, keyed by task name. */
export function getRunTasks(runKey) {
  return client.get(`/runs/${runKey}/tasks`).then(r => r.data)
}

/** Get a single task result for a run. */
export function getTask(runKey, taskName) {
  return client.get(`/runs/${runKey}/tasks/${taskName}`).then(r => r.data)
}

/** Get the figure index for a run. */
export function getRunFigures(runKey) {
  return client.get(`/runs/${runKey}/figures`).then(r => r.data)
}

/**
 * Compare a task's summary across multiple runs.
 * @param {string[]} runKeys - Array of run_key strings
 * @param {string} taskName  - Task to compare
 */
export function compareRuns(runKeys, taskName) {
  const params = new URLSearchParams()
  runKeys.forEach(k => params.append('run_keys', k))
  params.append('task_name', taskName)
  return client.get(`/runs/compare?${params}`).then(r => r.data)
}

/**
 * Get a sample of rows from the source dataset for a run.
 * @param {string} runKey
 * @param {number} n - number of rows (max 50)
 */
export function getRunSample(runKey, n = 10) {
  return client.get(`/runs/${runKey}/sample`, { params: { n } }).then(r => r.data)
}

/**
 * Get the data-health header bar summary for a run.
 * Returns { available, total_columns, categories } where categories is
 * keyed by dimension name and each value has { level, affected_count, pct_affected }.
 * If the scorer has not run, returns { available: false }.
 */
export function getDqStatus(runKey) {
  return client.get(`/runs/${runKey}/dq-status`).then(r => r.data)
}

// ── Relationships ────────────────────────────────────────────────────────────

/**
 * Get all pairwise associations for a run.
 * Returns { source, pairs, summary }
 */
export function getRunAssociations(runKey) {
  return client.get(`/runs/${runKey}/associations`).then(r => r.data)
}

/**
 * Get pairwise associations for a single column, sorted by abs metric desc.
 * @param {string} minStrength - optional: "strong" | "moderate" | "weak"
 */
export function getColumnAssociations(runKey, column, minStrength = '') {
  return client.get(
    `/runs/${runKey}/associations/${encodeURIComponent(column)}`,
    { params: minStrength ? { min_strength: minStrength } : {} }
  ).then(r => r.data)
}

/**
 * Fetch raw column values for pair plotting.
 * @param {string[]} cols - column names to fetch
 * @param {number}   maxRows - subsample cap (default 3000)
 * Returns { total_rows, sampled_rows, [colName]: [...values] }
 */
export function getColumnData(runKey, cols, maxRows = 3000) {
  const params = new URLSearchParams()
  cols.forEach(c => params.append('cols', c))
  params.append('max_rows', maxRows)
  return client.get(`/runs/${runKey}/column-data?${params}`).then(r => r.data)
}

// ── Figures ───────────────────────────────────────────────────────────────────

/**
 * Return the URL to fetch a figure file directly.
 * Use this as the src attribute on <img> tags or as the URL to load Plotly JSON.
 */
export function getColumnCorrelations(runKey, column, threshold = 0.0) {
  return client.get(`/runs/${runKey}/correlations/${encodeURIComponent(column)}`, {
    params: { threshold },
  }).then(r => r.data)
}

export function figureFileUrl(figureId) {
  return `/api/figures/${figureId}/file`
}
