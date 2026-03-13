<template>
  <div>
    <div class="page-header">
      <h1 class="page-title">{{ name }}</h1>
      <div class="search-wrap">
        <span class="search-icon">⌕</span>
        <input
          v-model="query"
          class="search-input"
          type="text"
          placeholder="Search runs…"
          autocomplete="off"
          spellcheck="false"
        />
        <button v-if="query" class="search-clear" @click="query = ''" title="Clear">✕</button>
      </div>
    </div>

    <div v-if="loading" class="loading">Loading runs…</div>
    <div v-else-if="error" class="error">{{ error }}</div>

    <template v-else>
      <div v-if="filtered.length === 0" class="empty-state">
        No runs match <strong>{{ query }}</strong>
      </div>

      <table v-else class="runs-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Date</th>
            <th>Depth</th>
            <th>Stage</th>
            <th>Rows</th>
            <th>Columns</th>
            <th>Quality</th>
            <th>Tasks</th>
            <th>Figures</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="run in filtered"
            :key="run.run_key"
            class="run-row"
            @click="$router.push(`/datasets/${name}/runs/${run.run_key}`)"
          >
            <td class="run-key" v-html="highlight(run.run_key, query)" />
            <td v-html="highlight(formatDate(run.ran_at), query)" />
            <td><span class="badge badge-blue">{{ run.profiling_depth }}</span></td>
            <td><span class="badge badge-gray">{{ run.inferred_stage ?? '-' }}</span></td>
            <td>{{ run.row_count?.toLocaleString() ?? '-' }}</td>
            <td>{{ run.col_count?.toLocaleString() ?? '-' }}</td>
            <td :class="qualityClass(run.quality_score)">
              {{ run.quality_score ?? '-' }}
            </td>
            <td>{{ run.task_count }}</td>
            <td>{{ run.fig_count }}</td>
          </tr>
        </tbody>
      </table>

      <div class="result-count" v-if="query">
        {{ filtered.length }} of {{ runs.length }} runs
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { formatDate, qualityClass } from '../utils.js'
import { listRuns } from '../api.js'

const props = defineProps({ name: String })

const runs    = ref([])
const loading = ref(true)
const error   = ref(null)
const query   = ref('')

onMounted(async () => {
  try {
    runs.value = await listRuns(props.name)
  } catch (e) {
    error.value = `Failed to load runs: ${e.message}`
  } finally {
    loading.value = false
  }
})

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return runs.value
  return runs.value.filter(run => {
    return (
      run.run_key.toLowerCase().includes(q)           ||
      (run.profiling_depth ?? '').toLowerCase().includes(q) ||
      (run.inferred_stage  ?? '').toLowerCase().includes(q) ||
      formatDate(run.ran_at).toLowerCase().includes(q)
    )
  })
})

function highlight(text, q) {
  if (!q.trim() || !text) return text ?? ''
  const escaped = q.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return String(text).replace(new RegExp(`(${escaped})`, 'gi'), '<mark>$1</mark>')
}
</script>

<style scoped>
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  gap: 16px;
  flex-wrap: wrap;
}

.page-title {
  font-size: 22px;
  font-weight: 600;
  color: #f1f5f9;
  text-transform: capitalize;
  margin: 0;
}

.search-wrap {
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: 10px;
  font-size: 16px;
  color: #64748b;
  pointer-events: none;
  line-height: 1;
}

.search-input {
  width: 240px;
  padding: 7px 32px 7px 30px;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  color: #e2e8f0;
  font-size: 14px;
  outline: none;
  transition: border-color 0.15s;
}

.search-input:focus { border-color: #60a5fa; }
.search-input::placeholder { color: #475569; }

.search-clear {
  position: absolute;
  right: 8px;
  background: none;
  border: none;
  color: #64748b;
  cursor: pointer;
  font-size: 12px;
  padding: 2px 4px;
  line-height: 1;
}
.search-clear:hover { color: #e2e8f0; }

.runs-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.runs-table th {
  text-align: left;
  padding: 10px 14px;
  color: #64748b;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  border-bottom: 1px solid #334155;
}

.runs-table td {
  padding: 12px 14px;
  border-bottom: 1px solid #1e293b;
  color: #e2e8f0;
}

.run-row {
  cursor: pointer;
  transition: background 0.12s;
}

.run-row:hover td { background: #1e293b; }

.run-key {
  font-family: monospace;
  font-size: 13px;
  color: #60a5fa;
}

.empty-state {
  color: #64748b;
  font-size: 14px;
  padding: 40px 0;
  text-align: center;
}

.empty-state strong { color: #94a3b8; }

.result-count {
  margin-top: 12px;
  font-size: 12px;
  color: #475569;
  text-align: right;
}

.score-excellent { color: #4ade80; font-weight: 600; }
.score-good      { color: #34d399; font-weight: 600; }
.score-warn      { color: #fb923c; font-weight: 600; }
.score-poor      { color: #f87171; font-weight: 600; }

:deep(mark) {
  background: #facc15;
  color: #0f172a;
  border-radius: 2px;
  padding: 0 1px;
}
</style>
