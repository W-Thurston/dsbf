<template>
  <div>
    <!-- Run header -->
    <div v-if="run" class="run-header card">
      <div class="header-metrics">
        <div class="metric">
          <span class="metric-label">Quality Score</span>
          <span class="metric-value" :class="qualityClass(run.quality_score)">
            {{ run.quality_score ?? '—' }}
          </span>
        </div>
        <div class="metric">
          <span class="metric-label">Rows</span>
          <span class="metric-value">{{ run.row_count?.toLocaleString() ?? '—' }}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Columns</span>
          <span class="metric-value">{{ run.col_count?.toLocaleString() ?? '—' }}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Depth</span>
          <span class="metric-value">{{ run.profiling_depth }}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Stage</span>
          <span class="metric-value">{{ run.inferred_stage ?? '—' }}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Run Date</span>
          <span class="metric-value">{{ formatDate(run.ran_at) }}</span>
        </div>
      </div>
    </div>

    <div v-if="loading" class="loading">Loading run…</div>
    <div v-else-if="error" class="error">{{ error }}</div>

    <template v-else>
      <!-- Tab bar -->
      <div class="tab-bar">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          class="tab-btn"
          :class="{ active: activeTab === tab.key }"
          @click="activeTab = tab.key"
        >
          {{ tab.label }}
        </button>
      </div>

      <!-- Tab content -->
      <div class="tab-content">
        <div v-if="activeTab === 'overview'">
          <p class="placeholder">Overview tab — coming soon.</p>
        </div>

        <div v-else-if="activeTab === 'distributions'">
          <p class="placeholder">Distributions tab — coming soon.</p>
        </div>

        <div v-else-if="activeTab === 'relationships'">
          <p class="placeholder">Relationships tab — coming soon.</p>
        </div>

        <div v-else-if="activeTab === 'quality'">
          <p class="placeholder">Quality tab — coming soon.</p>
        </div>

        <div v-else-if="activeTab === 'explore'">
          <p class="placeholder">Explore tab — coming soon.</p>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getRun, getRunTasks } from '../api.js'

const props = defineProps({
  name:   String,
  runKey: String,
})

const run     = ref(null)
const tasks   = ref(null)
const loading = ref(true)
const error   = ref(null)

const tabs = [
  { key: 'overview',       label: 'Overview'       },
  { key: 'distributions',  label: 'Distributions'  },
  { key: 'relationships',  label: 'Relationships'  },
  { key: 'quality',        label: 'Quality'        },
  { key: 'explore',        label: 'Explore'        },
]
const activeTab = ref('overview')

onMounted(async () => {
  try {
    // Fetch run metadata and all task results in parallel
    const [runData, taskData] = await Promise.all([
      getRun(props.runKey),
      getRunTasks(props.runKey),
    ])
    run.value   = runData
    tasks.value = taskData
  } catch (e) {
    error.value = `Failed to load run: ${e.message}`
  } finally {
    loading.value = false
  }
})

function formatDate(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function qualityClass(score) {
  if (score == null) return ''
  if (score >= 90) return 'score-excellent'
  if (score >= 70) return 'score-good'
  if (score >= 40) return 'score-warn'
  return 'score-poor'
}
</script>

<style scoped>
/* ── Run header ──────────────────────────────────────────────────────────── */
.run-header {
  margin-bottom: 20px;
}

.header-metrics {
  display: flex;
  gap: 32px;
  flex-wrap: wrap;
}

.metric {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metric-label {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.metric-value {
  font-size: 18px;
  font-weight: 600;
  color: #e2e8f0;
}

.score-excellent { color: #4ade80; }
.score-good      { color: #34d399; }
.score-warn      { color: #fb923c; }
.score-poor      { color: #f87171; }

/* ── Tab bar ─────────────────────────────────────────────────────────────── */
.tab-bar {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid #334155;
  margin-bottom: 20px;
}

.tab-btn {
  padding: 10px 18px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: #64748b;
  font-size: 14px;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
  margin-bottom: -1px;
}

.tab-btn:hover  { color: #e2e8f0; }
.tab-btn.active { color: #60a5fa; border-bottom-color: #60a5fa; }

/* ── Tab content ─────────────────────────────────────────────────────────── */
.tab-content { min-height: 200px; }

.placeholder {
  color: #475569;
  font-size: 14px;
  padding: 20px 0;
}

</style>
