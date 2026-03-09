<template>
  <div>
    <h1 class="page-title">{{ name }}</h1>

    <div v-if="loading" class="loading">Loading runs…</div>
    <div v-else-if="error" class="error">{{ error }}</div>

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
          v-for="run in runs"
          :key="run.run_key"
          class="run-row"
          @click="$router.push(`/datasets/${name}/runs/${run.run_key}`)"
        >
          <td class="run-key">{{ run.run_key }}</td>
          <td>{{ formatDate(run.ran_at) }}</td>
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
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { formatDate, qualityClass } from '../utils.js'
import { listRuns } from '../api.js'

const props = defineProps({ name: String })

const runs    = ref([])
const loading = ref(true)
const error   = ref(null)

onMounted(async () => {
  try {
    runs.value = await listRuns(props.name)
  } catch (e) {
    error.value = `Failed to load runs: ${e.message}`
  } finally {
    loading.value = false
  }
})

</script>

<style scoped>
.page-title {
  font-size: 22px;
  font-weight: 600;
  margin-bottom: 20px;
  color: #f1f5f9;
  text-transform: capitalize;
}

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

.score-excellent { color: #4ade80; font-weight: 600; }
.score-good      { color: #34d399; font-weight: 600; }
.score-warn      { color: #fb923c; font-weight: 600; }
.score-poor      { color: #f87171; font-weight: 600; }
</style>
