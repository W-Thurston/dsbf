<template>
  <div class="card col-meta-card">
    <div class="card-header">
      <span class="card-title">Dataset Metadata Overview</span>
      <input v-model="search" class="search-input" placeholder="Filter columns…" type="text" />
    </div>

    <div class="table-wrap">
      <div v-if="state === 'not_run'" class="es-not-run">
        Column metadata unavailable - <code>infer_types</code> did not run for this profiling depth.
      </div>
      <div v-else-if="state === 'empty'" class="es-empty">
        ✓ No columns found in this dataset.
      </div>
      <table v-else class="meta-table">
        <thead>
          <tr>
            <th class="col-frozen" @click="sortBy('column')" :class="sortClass('column')">
              Column <TooltipIcon text="The name of the column as it appears in the dataset." align="left" direction="down"/>
            </th>
            <th @click="sortBy('inferred')" :class="sortClass('inferred')">
              Inferred Type <TooltipIcon text="The raw data type inferred from the column values - e.g. int64, float64, object." direction="down"/>
            </th>
            <th @click="sortBy('intent')" :class="sortClass('intent')">
              Intent Type <TooltipIcon text="The analytical intent assigned to this column - how it should be treated during analysis (continuous, categorical, boolean, datetime, text)." direction="down"/>
            </th>
            <th @click="sortBy('null_pct_raw')" :class="sortClass('null_pct_raw')">
              Null % <TooltipIcon text="Percentage of rows where this column has a missing (null/NaN) value. Highlighted orange above 5%, red above 20%." direction="down"/>
            </th>
            <th @click="sortBy('unique_raw')" :class="sortClass('unique_raw')">
              Unique <TooltipIcon text="Number of distinct values in this column." direction="down"/>
            </th>
            <th @click="sortBy('mode')" :class="sortClass('mode')">
              Mode <TooltipIcon text="The most frequently occurring value in this column." direction="down"/>
            </th>
            <th @click="sortBy('mode_prop_raw')" :class="sortClass('mode_prop_raw')">
              Mode Proportion <TooltipIcon text="What fraction of all rows contain the most common value. A very high proportion (e.g. >90%) may indicate a near-constant or dominant column." direction="down"/>
            </th>
            <th @click="sortBy('cardinality_raw')" :class="sortClass('cardinality_raw')">
              Cardinality <TooltipIcon text="Number of unique values relative to total rows. High cardinality in a categorical column may indicate an ID or free-text field." direction="down"/>
            </th>
            <th @click="sortBy('skew_raw')" :class="sortClass('skew_raw')">
              Skew <TooltipIcon text="Statistical skewness of the distribution. Values beyond ±1 indicate asymmetry; beyond ±2 indicate heavy skew that may affect modelling." direction="down"/>
            </th>
            <th @click="sortBy('range')" :class="sortClass('range')">
              Range <TooltipIcon text="The [min, max] value range for numeric columns." align="right" direction="down"/>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in filteredRows" :key="row.column">
            <td class="col-frozen col-name" :title="row.column">{{ trunc(row.column, 24) }}</td>
            <td class="muted mono-sm">{{ row.inferred }}</td>
            <td><span class="type-badge" :class="intentClass(row.intent)">{{ row.intent }}</span></td>
            <td :class="nullClass(row.null_pct_raw)">{{ row.null_pct }}</td>
            <td>{{ row.unique }}</td>
            <td class="muted mono-sm" :title="row.mode">{{ trunc(row.mode, 18) }}</td>
            <td :class="propClass(row.mode_prop_raw)">{{ row.mode_prop }}</td>
            <td>{{ row.cardinality }}</td>
            <td :class="skewClass(row.skew_raw)">{{ row.skew }}</td>
            <td class="muted mono-sm" :title="row.range">{{ trunc(row.range, 22) }}</td>
          </tr>
        </tbody>
        </table>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'
import { trunc } from '../../utils.js'

const props = defineProps({
  tasks: { type: Object, default: () => ({}) },
})

const search  = ref('')
const sortKey = ref('column')
const sortDir = ref(1)

// ── Empty state ───────────────────────────────────────────────────────────────
const state = computed(() => {
  const types = props.tasks?.infer_types?.data
  if (!types) return 'not_run'
  if (Object.keys(types).length === 0) return 'empty'
  return 'ready'
})

function sortBy(key) {
  if (sortKey.value === key) { sortDir.value *= -1 }
  else { sortKey.value = key; sortDir.value = 1 }
}
function sortClass(key) {
  if (sortKey.value !== key) return 'sortable'
  return sortDir.value === 1 ? 'sort-asc' : 'sort-desc'
}

const rows = computed(() => {
  const types     = props.tasks.infer_types?.data                          ?? {}
  const nullPcts  = props.tasks.summarize_nulls?.data?.null_percentages    ?? {}
  const unique    = props.tasks.summarize_unique?.data                     ?? {}
  const skewness  = props.tasks.detect_skewness?.data                     ?? {}
  const numeric   = props.tasks.summarize_numeric?.data                   ?? {}
  const dominant  = props.tasks.detect_single_dominant_value?.data        ?? {}
  const valCounts = props.tasks.summarize_value_counts?.data              ?? {}

  return Object.entries(types).map(([col, typeInfo]) => {
    const null_pct_raw    = nullPcts[col]       ?? null
    const skew_raw        = skewness[col]       ?? null
    const numStats        = numeric[col]        ?? {}
    const domInfo         = dominant[col]       ?? {}
    const vc              = valCounts[col]      ?? {}
    const unique_raw      = typeof unique[col] === 'number' ? unique[col] : null
    const mode_prop_raw   = domInfo.mode_proportion ?? null
    const cardinality_raw = typeof vc === 'object' && !Array.isArray(vc)
      ? Object.keys(vc).length : unique_raw

    const minV = numStats.min
    const maxV = numStats.max
    const range = (minV != null && maxV != null)
      ? `[${+minV.toFixed(3)}, ${+maxV.toFixed(3)}]` : '-'

    return {
      column:           col,
      inferred:         typeInfo.inferred_dtype        ?? '-',
      intent:           typeInfo.analysis_intent_dtype ?? '-',
      null_pct_raw,
      null_pct:         null_pct_raw != null ? `${(null_pct_raw * 100).toFixed(1)}%` : '-',
      unique_raw,
      unique:           unique_raw != null ? unique_raw.toLocaleString() : '-',
      mode:             domInfo.mode != null ? String(domInfo.mode) : '-',
      mode_prop_raw,
      mode_prop:        mode_prop_raw != null ? `${(mode_prop_raw * 100).toFixed(1)}%` : '-',
      cardinality_raw,
      cardinality:      cardinality_raw != null ? cardinality_raw.toLocaleString() : '-',
      skew_raw,
      skew:             skew_raw != null ? skew_raw.toFixed(2) : '-',
      range,
    }
  })
})

const filteredRows = computed(() => {
  const q = search.value.toLowerCase()
  let out = q ? rows.value.filter(r => r.column.toLowerCase().includes(q)) : rows.value
  const k = sortKey.value
  const rawKeys = ['null_pct_raw', 'unique_raw', 'mode_prop_raw', 'cardinality_raw', 'skew_raw']
  return [...out].sort((a, b) => {
    const av = rawKeys.includes(k) ? (a[k] ?? -Infinity) : (a[k] ?? '')
    const bv = rawKeys.includes(k) ? (b[k] ?? -Infinity) : (b[k] ?? '')
    if (av < bv) return -1 * sortDir.value
    if (av > bv) return  1 * sortDir.value
    return 0
  })
})


function intentClass(intent) {
  return { continuous: 'intent-continuous', categorical: 'intent-categorical',
    boolean: 'intent-boolean', datetime: 'intent-datetime', text: 'intent-text',
  }[intent] ?? 'intent-unknown'
}
function nullClass(pct)  { return pct == null ? '' : pct >= 0.2 ? 'warn-high' : pct >= 0.05 ? 'warn-mid' : '' }
function skewClass(skew) { return skew == null ? '' : Math.abs(skew) > 1 ? 'warn-high' : '' }
function propClass(prop) { return prop == null ? '' : prop >= 0.9 ? 'warn-high' : prop >= 0.7 ? 'warn-mid' : '' }
</script>

<style scoped>
.col-meta-card { overflow: hidden; }

.es-not-run, .es-empty {
  font-size: 13px;
  padding: 32px 0;
  text-align: center;
}
.es-not-run { color: #475569; }
.es-empty   { color: #4ade80; }

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  flex-wrap: wrap;
  gap: 8px;
}

.search-input {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 13px;
  padding: 6px 10px;
  width: 180px;
  outline: none;
}
.search-input:focus { border-color: #60a5fa; }

.table-wrap {
  width: 100%;
  max-height: 420px;
  overflow: auto;
  /* Allow sticky headers to work and tooltips to escape the clip boundary */
  position: relative;
}

.meta-table {
  border-collapse: separate;
  border-spacing: 0;
  font-size: 13px;
  width: max-content;
}

.meta-table th {
  position: sticky;
  top: 0;
  z-index: 10;
  background: #1e293b;
  padding: 8px 14px;
  text-align: left;
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  border-bottom: 1px solid #334155;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}

/* Align tooltip icon inline with header text without breaking table layout */
.meta-table th :deep(.tip-wrap) {
  vertical-align: baseline;
  margin-left: 0px;
}

.col-frozen { position: sticky !important; left: 0; z-index: 2; background: #1e293b; }
th.col-frozen { z-index: 11; }
td.col-frozen { background: #1e293b; border-right: 1px solid #334155; }
.meta-table tr:hover td.col-frozen { background: #263548; }

.meta-table th.sort-asc::after  { content: ' ↑'; color: #60a5fa; }
.meta-table th.sort-desc::after { content: ' ↓'; color: #60a5fa; }
.meta-table th.sortable:hover   { color: #94a3b8; }

.meta-table td {
  padding: 8px 14px;
  border-bottom: 1px solid #0f172a;
  color: #e2e8f0;
  white-space: nowrap;
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.meta-table tr:hover td { background: #263548; }

.col-name { font-weight: 500; color: #f1f5f9; min-width: 140px; }
.muted    { color: #94a3b8; }
.mono-sm  { font-family: monospace; font-size: 12px; }

.type-badge {
  display: inline-block; padding: 2px 7px; border-radius: 8px;
  font-size: 11px; font-weight: 500; border: 1px solid; white-space: nowrap;
}
.intent-continuous  { background: #1e3a5f; color: #60a5fa; border-color: #60a5fa; }
.intent-categorical { background: #14291f; color: #4ade80; border-color: #4ade80; }
.intent-boolean     { background: #2d1b4e; color: #c084fc; border-color: #c084fc; }
.intent-datetime    { background: #3d2510; color: #fb923c; border-color: #fb923c; }
.intent-text        { background: #3d0f29; color: #f472b6; border-color: #f472b6; }
.intent-unknown     { background: #1e293b; color: #94a3b8; border-color: #475569; }

.warn-high { color: #f87171; font-weight: 600; }
.warn-mid  { color: #fb923c; }
</style>
