/**
 * src/api.js
 *
 * Thin wrapper around axios for all DSBF API endpoints.
 * Import and call these functions from Vue components — never
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

// ── Figures ───────────────────────────────────────────────────────────────────

/**
 * Return the URL to fetch a figure file directly.
 * Use this as the src attribute on <img> tags or as the URL to load Plotly JSON.
 */
export function figureFileUrl(figureId) {
  return `/api/figures/${figureId}/file`
}
