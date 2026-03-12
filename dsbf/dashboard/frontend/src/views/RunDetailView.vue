<template>
  <div>
    <!-- Quality Score header -->
    <div v-if="run" class="quality-header card">
      <div class="quality-title">Data Quality Score</div>
      <div class="quality-score" :class="qualityClass(run.quality_score)">
        {{ run.quality_score ?? '—' }}
      </div>
      <div class="quality-label" :class="qualityClass(run.quality_score)">
        {{ qualityLabel(run.quality_score) }}
      </div>
      <div class="quality-divider" />
      <div class="category-breakdown">
        <div v-for="(score, category) in categoryBreakdown" :key="category" class="category-item">
          <span class="category-label">
            {{ category.toUpperCase() }}
            <TooltipIcon :text="categoryTooltips[category] ?? category" align="center" />
          </span>
          <span class="category-value">{{ score }}</span>
        </div>
      </div>
    </div>

    <div v-if="loading" class="loading">Loading run…</div>
    <div v-else-if="error" class="error">{{ error }}</div>

    <template v-else>
      <!-- Tab bar -->
      <div class="tab-bar">
        <button
          v-for="tab in tabs" :key="tab.key"
          class="tab-btn" :class="{ active: activeTab === tab.key }"
          @click="activeTab = tab.key"
        >{{ tab.label }}</button>
        <div class="tab-spacer" />
        <div class="theme-toggle">
          <button class="theme-btn" :class="{ active: theme === 'light' }" @click="theme = 'light'">☀️ Light</button>
          <button class="theme-btn" :class="{ active: theme === 'dark' }"  @click="theme = 'dark'">🌙 Dark</button>
        </div>
      </div>

      <!-- Run metadata strip -->
      <div class="meta-strip card">
        <div class="meta-metric" v-for="m in runMetrics" :key="m.label">
          <span class="meta-label">
            {{ m.label }}
            <TooltipIcon :text="m.tooltip" align="center" />
          </span>
          <span class="meta-value" :title="m.fullValue ?? m.value" :data-key="m.dataKey">{{ m.value }}</span>
        </div>
      </div>

      <!-- Tab content -->
      <div class="tab-content">
        <OverviewTab       v-if="activeTab === 'overview'"       :run-key="runKey" :run="run" :tasks="tasks" :figures="figures" :theme="theme" />
        <DistributionsTab  v-else-if="activeTab === 'distributions'"  :run="run" :tasks="tasks" :figures="figures" :theme="theme" />
        <RelationshipsTab  v-else-if="activeTab === 'relationships'"  :run="run" :tasks="tasks" :figures="figures" :theme="theme" />
        <div v-else class="placeholder">
          {{ activeTab.charAt(0).toUpperCase() + activeTab.slice(1) }} tab — coming soon.
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { getRun, getRunTasks, getRunFigures } from '../api.js'
import OverviewTab        from './tabs/OverviewTab.vue'
import DistributionsTab   from './tabs/DistributionsTab.vue'
import RelationshipsTab   from './tabs/RelationshipsTab.vue'
import TooltipIcon  from '../components/TooltipIcon.vue'
import { formatDate, qualityClass, qualityLabel } from '../utils.js'

const props = defineProps({ name: String, runKey: String })

const run     = ref(null)
const tasks   = ref({})
const figures = ref([])
const loading = ref(true)
const error   = ref(null)
const theme   = ref('dark')

watch(theme, (val) => {
  document.body.classList.toggle('theme-light', val === 'light')
  document.body.classList.toggle('theme-dark',  val === 'dark')
}, { immediate: true })

const tabs = [
  { key: 'overview',      label: 'Overview'      },
  { key: 'distributions', label: 'Distributions' },
  { key: 'relationships', label: 'Relationships' },
  { key: 'quality',       label: 'Quality'       },
  { key: 'explore',       label: 'Explore'       },
]
const activeTab = ref('overview')

async function loadRun(runKey) {
  loading.value = true
  error.value   = null
  run.value     = null
  tasks.value   = {}
  figures.value = []
  try {
    const [runData, taskData, figureData] = await Promise.all([
      getRun(runKey),
      getRunTasks(runKey),
      getRunFigures(runKey),
    ])
    run.value     = runData
    tasks.value   = taskData
    figures.value = figureData
  } catch (e) {
    error.value = `Failed to load run: ${e.message}`
  } finally {
    loading.value = false
  }
}

onMounted(() => loadRun(props.runKey))
watch(() => props.runKey, (key) => { if (key) loadRun(key) })

const categoryBreakdown = computed(() =>
  tasks.value?.data_quality_scorer?.summary?.category_breakdown ?? {}
)

const categoryTooltips = {
  completeness: 'Measures how much data is present vs missing. A high score means few null or empty values across columns.',
  consistency:  'Checks whether values follow expected patterns and types — e.g. no text in numeric columns, valid date formats.',
  distribution: 'Evaluates whether column distributions look reasonable — flags extreme skewness, dominant values, or unusual spreads.',
  redundancy:   'Detects duplicate columns, constant columns, or features that carry identical information.',
  drift:        'Compares this run\'s statistics against previous runs to surface unexpected shifts in the data.',
}

const runMetrics = computed(() => {
  if (!run.value) return []
  const shape   = tasks.value?.summarize_dataset_shape?.data ?? {}
  const dupCount = tasks.value?.detect_duplicates?.data?.duplicate_count ?? null
  const rowCount = run.value.row_count ?? null

  let dupValue = '—'
  if (dupCount != null) {
    const pct = rowCount ? ` (${((dupCount / rowCount) * 100).toFixed(1)}%)` : ''
    dupValue  = `${dupCount.toLocaleString()}${pct}`
  }

  return [
    {
      label:   'Rows',
      tooltip: 'Total number of rows (observations) in the dataset.',
      value:   rowCount?.toLocaleString() ?? '—',
    },
    {
      label:   'Columns',
      tooltip: 'Total number of columns (features) in the dataset.',
      value:   run.value.col_count?.toLocaleString() ?? '—',
    },
    {
      label:   'Memory',
      tooltip: 'Approximate memory footprint of the dataset when loaded into a pandas DataFrame.',
      value:   shape.approx_memory_MB != null ? `${shape.approx_memory_MB} MB` : '—',
    },
    {
      label:   'Missing',
      tooltip: 'Percentage of all cells in the dataset that contain a null or missing value.',
      value:   shape.null_cell_percentage != null
        ? `${(shape.null_cell_percentage * 100).toFixed(1)}%` : '—',
    },
    {
      label:   'Duplicate Rows',
      tooltip: 'Number of rows that are exact duplicates of another row, with their percentage of the total. Duplicates can inflate counts and skew model training.',
      value:   dupValue,
      dataKey: 'dup',
    },
    {
      label:   'Depth',
      tooltip: 'Profiling depth used for this run: basic (fast, core stats), standard (recommended), or full (all tasks including expensive checks).',
      value:   run.value.profiling_depth ?? '—',
    },
    {
      label:   'Stage',
      tooltip: 'Inferred lifecycle stage of the dataset — e.g. raw (unprocessed), exploratory, or modelling-ready. Affects which checks are prioritised.',
      value:   run.value.inferred_stage ?? '—',
    },
  ]
})

</script>

<style scoped>
.quality-header { text-align: center; margin-bottom: 0; padding: 24px 32px 20px; }
.quality-title  { font-size: 15px; color: #94a3b8; margin-bottom: 6px; }
.quality-score  { font-size: 56px; font-weight: 800; line-height: 1; margin-bottom: 4px; }
.quality-label  { font-size: 14px; margin-bottom: 16px; }
.quality-divider { height: 1px; background: #334155; margin: 0 -32px 16px; }

.category-breakdown {
  display: flex;
  justify-content: space-around;
  flex-wrap: wrap;
  width: 100%;
}
.category-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  flex: 1;
}
.category-label {
  font-size: 11px;
  color: #64748b;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
  gap: 2px;
}
.category-value { font-size: 18px; font-weight: 700; color: #e2e8f0; }

.score-excellent { color: #4ade80; }
.score-good      { color: #34d399; }
.score-warn      { color: #fb923c; }
.score-poor      { color: #f87171; }

/* Tab bar */
.tab-bar {
  display: flex;
  align-items: center;
  gap: 4px;
  border-bottom: 1px solid #334155;
  margin: 16px 0;
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
  white-space: nowrap;
}
.tab-btn:hover  { color: #e2e8f0; }
.tab-btn.active { color: #60a5fa; border-bottom-color: #60a5fa; }
.tab-spacer     { flex: 1; }
.theme-toggle   { display: flex; gap: 4px; padding-bottom: 4px; }
.theme-btn {
  padding: 5px 12px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  color: #64748b;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}
.theme-btn.active { background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }

/* Meta strip */
.meta-strip {
  display: flex;
  justify-content: space-around;
  flex-wrap: wrap;
  padding: 16px 24px;
  margin-bottom: 16px;
}
.meta-metric {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  flex: 1;
  padding: 8px 16px;
  border-right: 1px solid #334155;
  min-width: 0;
}
.meta-metric:last-child { border-right: none; }
.meta-label {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  text-align: center;
  display: flex;
  align-items: center;
  gap: 2px;
}
.meta-value {
  font-size: 18px;
  font-weight: 600;
  color: #f1f5f9;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
  cursor: default;
}

/* Duplicate rows value includes a percentage — allow slightly smaller font
   so it stays on one line without truncating on typical screen widths */
.meta-metric:has(.meta-value[data-key="dup"]) .meta-value {
  font-size: 15px;
}

.tab-content { min-height: 200px; }
.placeholder { color: #475569; font-size: 14px; padding: 40px 0; text-align: center; }
</style>
