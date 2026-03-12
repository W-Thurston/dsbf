<template>
  <div class="card sample-card">
    <div class="card-header">
      <span class="card-title">
        Dataset Sample
        <TooltipIcon
          text="The first rows of the source dataset as loaded from disk. Useful for a quick sanity check on raw values before any transformations."
          align="right"
        />
      </span>
      <div class="header-controls">
        <label class="n-label">Rows:</label>
        <select v-model="nRows" class="n-select" @change="fetchSample">
          <option :value="5">5</option>
          <option :value="10">10</option>
          <option :value="25">25</option>
          <option :value="50">50</option>
        </select>
      </div>
    </div>

    <div v-if="loading" class="loading">Loading sample…</div>
    <div v-else-if="error === 'no_source_path'" class="sample-unavailable">
      <span class="unavail-icon">ⓘ</span>
      Source path not recorded for this run. Re-run the profiler to enable data sampling.
    </div>
    <div v-else-if="error === 'no_data'" class="sample-unavailable">
      <span class="unavail-icon">ⓘ</span>
      Data could not be loaded for this run.
    </div>
    <div v-else-if="error" class="sample-error">{{ error }}</div>
    <div v-else-if="!columns.length" class="loading">No sample data available.</div>

    <div v-else class="table-wrap">
      <table class="sample-table">
        <thead>
          <tr>
            <th class="row-idx">#</th>
            <th v-for="col in columns" :key="col" :title="col">{{ trunc(col, 18) }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in rows" :key="i">
            <td class="row-idx muted">{{ i + 1 }}</td>
            <td
              v-for="col in columns"
              :key="col"
              :title="row[col] != null ? String(row[col]) : ''"
              :class="{ 'null-cell': row[col] == null }"
            >{{ row[col] != null ? trunc(String(row[col]), 20) : 'null' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'
import { getRunSample } from '../../api.js'
import { trunc } from '../../utils.js'

const props = defineProps({
  runKey: { type: String, required: true },
  run:    { type: Object, default: null },
})

const loading = ref(true)
const error   = ref(null)
const columns = ref([])
const rows    = ref([])
const nRows   = ref(10)

async function fetchSample() {
  if (!props.runKey) return

  // Reset stale data immediately so the previous run's sample doesn't linger
  columns.value = []
  rows.value    = []
  error.value   = null
  loading.value = true

  // For built-in datasets (seaborn/sklearn) source_path is null but the backend
  // can still load data via _load_dataframe. Only bail early for runs that
  // pre-date path tracking — identified by dataset_source being unknown and
  // source_path being null.
  if (props.run && props.run.run_key === props.runKey) {
    const src    = props.run.source_path
    const source = props.run.dataset_source ?? 'unknown'
    if (!src && source === 'unknown') {
      loading.value = false
      error.value   = 'no_source_path'
      return
    }
  }
  try {
    const data    = await getRunSample(props.runKey, nRows.value)
    if (data.error) throw new Error(data.error)
    columns.value = data.columns ?? []
    rows.value    = data.rows    ?? []
  } catch (e) {
    error.value = `Could not load sample: ${e.message}`
  } finally {
    loading.value = false
  }
}

onMounted(fetchSample)
watch(() => props.runKey, () => fetchSample(), { immediate: false })
// Re-run check when run arrives in case we need to show the no_source_path message
watch(() => props.run?.run_key, (key) => { if (key === props.runKey) fetchSample() })
</script>

<style scoped>
.sample-card { overflow: hidden; height: 420px; display: flex; flex-direction: column; }

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  flex-wrap: wrap;
  gap: 8px;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 6px;
}

.n-label { font-size: 12px; color: #64748b; }

.n-select {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 13px;
  padding: 4px 8px;
  outline: none;
  cursor: pointer;
}
.n-select:focus { border-color: #60a5fa; }

.table-wrap { overflow-x: auto; overflow-y: auto; flex: 1; min-height: 0; }

.sample-table {
  border-collapse: separate;
  border-spacing: 0;
  font-size: 12px;
  width: max-content;
  min-width: 100%;
}

.sample-table th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: #1e293b;
  padding: 7px 12px;
  text-align: left;
  color: #64748b;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  border-bottom: 1px solid #334155;
  white-space: nowrap;
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: default;
}

.sample-table td {
  padding: 6px 12px;
  border-bottom: 1px solid #0f172a;
  color: #e2e8f0;
  white-space: nowrap;
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: monospace;
  font-size: 12px;
}

.sample-table tr:hover td { background: #263548; }

.row-idx {
  color: #334155 !important;
  font-variant-numeric: tabular-nums;
  min-width: 32px;
  user-select: none;
}

.null-cell { color: #475569 !important; font-style: italic; }

.loading { color: #94a3b8; font-size: 13px; padding: 24px 0; text-align: center; }
.sample-error { color: #f87171; font-size: 13px; padding: 16px 0; }

.muted { color: #475569; }

.sample-unavailable {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #475569;
  font-size: 13px;
  padding: 24px 0;
  text-align: center;
  justify-content: center;
}
.unavail-icon { font-size: 16px; color: #334155; }
</style>
