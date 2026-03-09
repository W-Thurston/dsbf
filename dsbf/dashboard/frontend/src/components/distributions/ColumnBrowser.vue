<template>
  <div class="col-browser">
    <div class="browser-search">
      <input
        v-model="search"
        class="browser-search-input"
        placeholder="Filter columns…"
        type="text"
      />
    </div>

    <!-- Legend -->
    <div class="browser-legend">
      <span class="legend-item">
        <span class="legend-warn">●</span> Has alerts
      </span>
      <span class="legend-item">
        <span class="legend-bar-wrap"><span class="legend-bar-fill" /></span> Null %
      </span>
    </div>

    <div class="browser-list">
      <template v-for="group in filteredGroups" :key="group.intent">
        <div class="group-header">
          <span class="group-badge" :class="`intent-${group.intent}`">{{ group.intent }}</span>
          <span class="group-count">{{ group.columns.length }}</span>
        </div>
        <button
          v-for="col in group.columns"
          :key="col.name"
          class="col-row"
          :class="{ active: col.name === selected }"
          @click="$emit('select', col.name)"
        >
          <span class="col-name" :title="col.name">{{ col.name }}</span>
          <span class="col-indicators">
            <span v-if="col.hasWarning" class="warn-dot" title="Has data quality alerts">●</span>
            <span
              class="null-bar-wrap"
              :title="`Null: ${(col.nullPct * 100).toFixed(1)}%`"
            >
              <span class="null-bar" :style="{ width: `${Math.min(col.nullPct * 100, 100)}%` }" />
            </span>
          </span>
        </button>
      </template>

      <div v-if="filteredGroups.length === 0" class="no-cols">No columns match.</div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  tasks:    { type: Object, default: () => ({}) },
  selected: { type: String, default: null },
})
defineEmits(['select'])

const search = ref('')

const INTENT_ORDER = ['continuous', 'categorical', 'boolean', 'datetime', 'text', 'unknown']

const columns = computed(() => {
  const types     = props.tasks.infer_types?.data                        ?? {}
  const nullPcts  = props.tasks.summarize_nulls?.data?.null_percentages  ?? {}
  const skewness  = props.tasks.detect_skewness?.data                    ?? {}
  const hiCard    = props.tasks.detect_high_cardinality?.data            ?? {}
  const constant  = new Set(props.tasks.detect_constant_columns?.data?.constant_columns ?? [])
  const idCols    = props.tasks.detect_id_columns?.data                  ?? {}
  const oob       = props.tasks.detect_out_of_bounds?.data               ?? {}
  const zeroPcts  = props.tasks.detect_zeros?.data?.zero_percentages     ?? {}
  const dom       = props.tasks.detect_single_dominant_value?.data       ?? {}
  const numeric   = props.tasks.summarize_numeric?.data                  ?? {}
  const vcData    = props.tasks.summarize_value_counts?.data             ?? {}
  const rowCount  = props.tasks.summarize_dataset_shape?.data?.num_rows  ?? null
  const unique    = props.tasks.summarize_unique?.data                   ?? {}

  return Object.entries(types).map(([name, typeInfo]) => {
    const nullPct  = nullPcts[name]  ?? 0
    const skew     = skewness[name]  ?? null
    const nm       = numeric[name]   ?? {}
    const domCol   = dom[name]       ?? {}
    const intent   = typeInfo.analysis_intent_dtype ?? 'unknown'

    // Mirror every non-"good" branch in ColumnGuidanceCard
    let hasWarning = false

    // Universal
    if (constant.has(name))     hasWarning = true
    if (nullPct >= 0.05)        hasWarning = true

    if (intent === 'continuous') {
      if (name in idCols)                               hasWarning = true
      if (skew !== null && Math.abs(skew) > 1)          hasWarning = true
      if (nm.near_zero_variance === true)               hasWarning = true
      if (name in oob)                                  hasWarning = true
      if ((zeroPcts[name] ?? 0) > 0.3)                  hasWarning = true
      // mean/median divergence
      const mean = nm.mean ?? null, median = nm['50%'] ?? null, std = nm.std ?? null
      if (mean != null && median != null && std != null && std > 0) {
        if (Math.abs(mean - median) / std > 0.5)        hasWarning = true
      }
    }

    if (intent === 'categorical' || intent === 'text') {
      const cardRatio = (unique[name] != null && rowCount != null)
        ? unique[name] / rowCount : null
      if (name in idCols || (cardRatio != null && cardRatio > 0.9)) hasWarning = true
      if (name in hiCard)                               hasWarning = true
      if ((domCol.mode_proportion ?? 0) >= 0.7)         hasWarning = true
    }

    if (intent === 'boolean') {
      const vc    = vcData[name] ?? {}
      const vals  = Object.values(vc)
      const total = vals.reduce((s, n) => s + n, 0)
      const topN  = Math.max(...vals, 0)
      if (total > 0 && topN / total > 0.75)             hasWarning = true
    }
    return {
      name,
      intent:     typeInfo.analysis_intent_dtype ?? 'unknown',
      nullPct,
      hasWarning,
    }
  })
})

const filteredGroups = computed(() => {
  const q    = search.value.toLowerCase()
  const cols = q ? columns.value.filter(c => c.name.toLowerCase().includes(q)) : columns.value

  const byIntent = {}
  for (const col of cols) {
    if (!byIntent[col.intent]) byIntent[col.intent] = []
    byIntent[col.intent].push(col)
  }

  return INTENT_ORDER
    .filter(intent => byIntent[intent]?.length)
    .map(intent => ({ intent, columns: byIntent[intent] }))
})
</script>

<style scoped>
.col-browser {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.browser-search { padding: 0 0 10px 0; flex-shrink: 0; }

.browser-search-input {
  width: 100%;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 12px;
  padding: 6px 10px;
  outline: none;
  box-sizing: border-box;
}
.browser-search-input:focus { border-color: #60a5fa; }

.browser-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 4px 4px;
  position: sticky;
  top: 0;
  background: #1e293b;
  z-index: 1;
}

.group-badge {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 2px 7px;
  border-radius: 8px;
  border: 1px solid;
}
.intent-continuous  { background: #1e3a5f; color: #60a5fa; border-color: #60a5fa; }
.intent-categorical { background: #14291f; color: #4ade80; border-color: #4ade80; }
.intent-boolean     { background: #2d1b4e; color: #c084fc; border-color: #c084fc; }
.intent-datetime    { background: #3d2510; color: #fb923c; border-color: #fb923c; }
.intent-text        { background: #3d0f29; color: #f472b6; border-color: #f472b6; }
.intent-unknown     { background: #1e293b; color: #94a3b8; border-color: #475569; }

.group-count {
  font-size: 11px;
  color: #475569;
}

.col-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 8px;
  border-radius: 6px;
  border: 1px solid transparent;
  background: transparent;
  color: #94a3b8;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
  width: 100%;
  gap: 6px;
  transition: background 0.1s, color 0.1s;
}
.col-row:hover  { background: #263548; color: #e2e8f0; }
.col-row.active { background: #1e3a5f; color: #e2e8f0; border-color: #334d6e; }

.col-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: monospace;
}

.col-indicators {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-shrink: 0;
}

.browser-legend {
  display: flex;
  gap: 12px;
  padding: 4px 2px 8px;
  border-bottom: 1px solid #1e293b;
  margin-bottom: 4px;
  flex-shrink: 0;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  color: #475569;
  white-space: nowrap;
}

.legend-warn { color: #fb923c; font-size: 9px; }

.legend-bar-wrap {
  width: 24px;
  height: 4px;
  background: #1e293b;
  border-radius: 2px;
  overflow: hidden;
  flex-shrink: 0;
}
.legend-bar-fill {
  display: block;
  width: 60%;
  height: 100%;
  background: #f87171;
  border-radius: 2px;
}

.warn-dot { color: #fb923c; font-size: 8px; }

.null-bar-wrap {
  width: 30px;
  height: 4px;
  background: #1e293b;
  border-radius: 2px;
  overflow: hidden;
  flex-shrink: 0;
}
.null-bar {
  height: 100%;
  background: #f87171;
  border-radius: 2px;
  transition: width 0.2s;
}

.no-cols {
  color: #475569;
  font-size: 12px;
  padding: 16px 8px;
  text-align: center;
}
</style>
