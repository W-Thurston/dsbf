<template>
  <div class="card correlations-card">
    <div class="card-title">
      Top Correlations
      <TooltipIcon
        text="Pearson correlation with other numeric columns. Values near ±1 indicate a strong linear relationship. Strong correlations (>0.9) may indicate data leakage or redundant features."
        direction="down"
        align="left"
      />
    </div>

    <div v-if="loading" class="corr-loading">Loading correlations…</div>

    <div v-else-if="unavailable" class="corr-unavailable">
      <span class="unavail-icon">📊</span>
      <span>{{ unavailableReason }}</span>
    </div>

    <div v-else-if="!rows.length" class="corr-empty">
      No correlations above threshold for this column.
    </div>

    <div v-else class="corr-body">
      <!-- Strength filter -->
      <div class="corr-filters">
        <button
          v-for="f in filters"
          :key="f.key"
          class="filter-chip"
          :class="{ active: activeFilter === f.key }"
          @click="activeFilter = f.key"
        >{{ f.label }}</button>
      </div>

      <div class="corr-list">
        <div
          v-for="row in visibleRows"
          :key="row.column"
          class="corr-row"
          :class="strengthClass(row.correlation)"
          @click="$emit('selectColumn', row.column)"
          :title="`Navigate to ${row.column}`"
        >
          <span class="corr-col-name">{{ row.column }}</span>
          <div class="corr-bar-wrap">
            <!-- negative side -->
            <div class="corr-bar-neg-track">
              <div
                v-if="row.correlation < 0"
                class="corr-bar-neg-fill"
                :style="{ width: `${Math.abs(row.correlation) * 100}%` }"
              />
            </div>
            <!-- midpoint marker -->
            <div class="corr-midpoint" />
            <!-- positive side -->
            <div class="corr-bar-pos-track">
              <div
                v-if="row.correlation > 0"
                class="corr-bar-pos-fill"
                :style="{ width: `${row.correlation * 100}%` }"
              />
            </div>
          </div>
          <span class="corr-value" :class="valueClass(row.correlation)">
            {{ row.correlation >= 0 ? '+' : '' }}{{ row.correlation.toFixed(3) }}
          </span>
        </div>
      </div>

      <div v-if="truncated" class="corr-truncated">
        Showing {{ visibleRows.length }} of {{ rows.length }} correlated columns.
        <button class="show-more-btn" @click="showAll = !showAll">
          {{ showAll ? 'Show fewer' : 'Show all' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'
import { getColumnCorrelations } from '../../api.js'

const props = defineProps({
  runKey: { type: String, required: true },
  column: { type: String, required: true },
})

defineEmits(['selectColumn'])

const loading          = ref(false)
const unavailable      = ref(false)
const unavailableReason = ref('')
const allCorrelations  = ref([])
const activeFilter     = ref('all')
const showAll          = ref(false)

const DEFAULT_MAX = 15

const filters = [
  { key: 'all',      label: 'All'          },
  { key: 'strong',   label: 'Strong (≥0.7)' },
  { key: 'moderate', label: 'Moderate (≥0.4)' },
]

async function load() {
  if (!props.runKey || !props.column) return
  loading.value     = true
  unavailable.value = false
  allCorrelations.value = []
  try {
    const result = await getColumnCorrelations(props.runKey, props.column, 0.1)
    if (result.unavailable) {
      unavailable.value       = true
      unavailableReason.value = result.reason ?? 'Correlation data unavailable.'
    } else {
      allCorrelations.value = result.correlations ?? []
    }
  } catch {
    unavailable.value       = true
    unavailableReason.value = 'Could not load correlations.'
  } finally {
    loading.value = false
  }
}

watch(() => [props.runKey, props.column], load, { immediate: true })

const rows = computed(() => {
  const threshold = activeFilter.value === 'strong'
    ? 0.7
    : activeFilter.value === 'moderate'
    ? 0.4
    : 0.1
  return allCorrelations.value.filter(r => Math.abs(r.correlation) >= threshold)
})

const truncated    = computed(() => !showAll.value && rows.value.length > DEFAULT_MAX)
const visibleRows  = computed(() =>
  truncated.value ? rows.value.slice(0, DEFAULT_MAX) : rows.value
)

function strengthClass(val) {
  const abs = Math.abs(val)
  if (abs >= 0.9) return 'strength-critical'
  if (abs >= 0.7) return 'strength-strong'
  if (abs >= 0.4) return 'strength-moderate'
  return 'strength-weak'
}

function valueClass(val) {
  const abs = Math.abs(val)
  if (abs >= 0.9) return val < 0 ? 'val-neg-critical' : 'val-pos-critical'
  if (abs >= 0.7) return val < 0 ? 'val-neg-strong'   : 'val-pos-strong'
  return val < 0 ? 'val-neg' : 'val-pos'
}
</script>

<style scoped>
.correlations-card {
  display: flex;
  flex-direction: column;
}

.corr-loading,
.corr-empty {
  font-size: 13px;
  color: #475569;
  padding: 20px 0;
  text-align: center;
}

.corr-unavailable {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: 12px;
  color: #475569;
  padding: 12px 0;
  line-height: 1.5;
}
.unavail-icon { flex-shrink: 0; }

/* Filters */
.corr-filters {
  display: flex;
  gap: 6px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.filter-chip {
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 10px;
  border: 1px solid #334155;
  background: transparent;
  color: #64748b;
  cursor: pointer;
  transition: all 0.15s;
}
.filter-chip:hover  { background: #1e293b; color: #94a3b8; }
.filter-chip.active { background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }

/* Correlation rows */
.corr-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.corr-row {
  display: grid;
  grid-template-columns: 140px 1fr 60px;
  align-items: center;
  gap: 8px;
  padding: 4px 6px;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.1s;
}
.corr-row:hover { background: #1e293b; }

.corr-col-name {
  font-size: 11px;
  font-family: monospace;
  color: #94a3b8;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.corr-row:hover .corr-col-name { color: #e2e8f0; }

/* Bidirectional bar */
.corr-bar-wrap {
  display: flex;
  align-items: center;
  gap: 0;
  height: 10px;
}

.corr-bar-neg-track {
  flex: 1;
  height: 6px;
  background: #0f172a;
  border-radius: 3px 0 0 3px;
  display: flex;
  justify-content: flex-end;
  overflow: hidden;
}
.corr-bar-neg-fill {
  height: 100%;
  background: #60a5fa;
  border-radius: 3px 0 0 3px;
  transition: width 0.3s;
}

.corr-midpoint {
  width: 2px;
  height: 10px;
  background: #334155;
  flex-shrink: 0;
}

.corr-bar-pos-track {
  flex: 1;
  height: 6px;
  background: #0f172a;
  border-radius: 0 3px 3px 0;
  overflow: hidden;
}
.corr-bar-pos-fill {
  height: 100%;
  background: #f87171;
  border-radius: 0 3px 3px 0;
  transition: width 0.3s;
}

/* Strength row tinting */
.strength-critical { background: rgba(248, 113, 113, 0.05); }
.strength-critical:hover { background: rgba(248, 113, 113, 0.12) !important; }

.corr-value {
  font-size: 11px;
  font-family: monospace;
  font-weight: 600;
  text-align: right;
  white-space: nowrap;
}
.val-pos-critical { color: #f87171; }
.val-pos-strong   { color: #fb923c; }
.val-pos          { color: #94a3b8; }
.val-neg-critical { color: #60a5fa; }
.val-neg-strong   { color: #818cf8; }
.val-neg          { color: #94a3b8; }

.corr-truncated {
  margin-top: 10px;
  font-size: 11px;
  color: #475569;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.show-more-btn {
  font-size: 11px;
  color: #60a5fa;
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
}
.show-more-btn:hover { text-decoration: underline; }
</style>
