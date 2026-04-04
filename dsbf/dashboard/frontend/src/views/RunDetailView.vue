<template>
  <div>
    <!-- Data health bar -->
    <DataHealthBar v-if="run" :run-key="runKey" :active-tab="activeTab" />

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

      <!-- Tab content -->
      <div class="tab-content">
        <OverviewTab       v-if="activeTab === 'overview'"            :run-key="runKey" :run="run" :tasks="tasks" :figures="figures" :theme="theme" />
        <DistributionsTab  v-else-if="activeTab === 'distributions'"  :run="run" :tasks="tasks" :figures="figures" :theme="theme" />
        <RelationshipsTab  v-else-if="activeTab === 'relationships'"  :run="run" :tasks="tasks" :figures="figures" :theme="theme" />
        <QualityTab        v-else-if="activeTab === 'quality'"        :run-key="runKey" :tasks="tasks" />
        <TimeSeriesTab     v-else-if="activeTab === 'time_series'"    :tasks="tasks" :theme="theme" />
        <MlReadinessTab    v-else-if="activeTab === 'ml_readiness'"   :run-key="runKey" />
        <div v-else class="placeholder">
          {{ activeTab.charAt(0).toUpperCase() + activeTab.slice(1) }} tab - coming soon.
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { getRun, getRunTasks, getRunFigures } from '../api.js'
import OverviewTab        from './tabs/OverviewTab.vue'
import DistributionsTab   from './tabs/DistributionsTab.vue'
import RelationshipsTab   from './tabs/RelationshipsTab.vue'
import QualityTab         from './tabs/QualityTab.vue'
import MlReadinessTab    from './tabs/MlReadinessTab.vue'
import TimeSeriesTab      from './tabs/TimeSeriesTab.vue'
import DataHealthBar      from '../components/DataHealthBar.vue'

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
  { key: 'quality',       label: 'Quality'       },
  { key: 'distributions', label: 'Distributions' },
  { key: 'relationships', label: 'Relationships' },
  { key: 'time_series',   label: 'Time Series'   },
  { key: 'ml_readiness',  label: 'ML Readiness'  },
  { key: 'column_detail', label: 'Column Detail' },
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

</script>

<style scoped>
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

.tab-content { min-height: 200px; }
.placeholder { color: #64748b; font-size: 14px; padding: 40px 0; text-align: center; }
</style>
