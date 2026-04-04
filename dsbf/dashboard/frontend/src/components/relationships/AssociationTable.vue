<template>
  <div class="assoc-table-wrap">
    <!-- Controls row -->
    <div class="controls-row">
      <div class="search-wrap">
        <span class="search-icon">⌕</span>
        <input v-model="search" class="search-input" placeholder="Filter columns…" autocomplete="off" />
        <button v-if="search" class="search-clear" @click="search = ''">✕</button>
      </div>

      <div class="filter-chips">
        <span class="chip-label">Intent:</span>
        <button
          v-for="t in intentOptions" :key="t.value"
          class="chip" :class="{ active: intentFilter === t.value }"
          @click="intentFilter = t.value"
        >{{ t.label }}</button>
      </div>

      <div class="filter-chips">
        <span class="chip-label">Strength:</span>
        <button
          v-for="s in strengthOptions" :key="s.value"
          class="chip" :class="{ active: strengthFilter === s.value }"
          @click="strengthFilter = s.value"
        >{{ s.label }}</button>
      </div>

      <div class="sort-wrap">
        <span class="chip-label">Sort:</span>
        <button
          v-for="s in sortOptions" :key="s.value"
          class="chip" :class="{ active: sortBy === s.value }"
          @click="sortBy = s.value"
        >{{ s.label }}</button>
      </div>
    </div>

    <!-- Table -->
    <div v-if="filtered.length === 0" class="empty">
      No associations match the current filters.
    </div>

    <table v-else class="assoc-table">
      <thead>
        <tr>
          <th>Column</th>
          <th>Type</th>
          <th class="th-metric">Association</th>
          <th>Strength</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="row in filtered"
          :key="row.column"
          class="assoc-row"
          :class="{ selected: modelValue === row.column }"
          @click="$emit('update:modelValue', row.column)"
        >
          <td class="td-name">{{ row.column }}</td>
          <td class="td-type">
            <span class="intent-badge" :class="`intent-${row.col_other_intent}`">
              {{ row.col_other_intent }}
            </span>
          </td>
          <td class="td-metric">
            <div class="metric-cell">
              <span class="metric-type-label">{{ metricLabel(row.metric_type) }}</span>
              <span class="metric-val" :class="metricClass(row.metric, row.metric_type)">
                {{ formatMetric(row.metric, row.metric_type) }}
              </span>
            </div>
          </td>
          <td class="td-strength">
            <span class="strength-badge" :class="`str-${row.strength}`">
              {{ row.strength }}
            </span>
          </td>
        </tr>
      </tbody>
    </table>

    <div class="result-count" v-if="search || intentFilter !== 'all' || strengthFilter !== 'all'">
      {{ filtered.length }} of {{ rows.length }} columns
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  rows:       { type: Array,  default: () => [] }, // from getColumnAssociations
  modelValue: { type: String, default: null },      // selected secondary column
})
const emit = defineEmits(['update:modelValue'])

const search         = ref('')
const intentFilter   = ref('all')
const strengthFilter = ref('all')
const sortBy         = ref('metric')

const intentOptions = [
  { value: 'all',        label: 'All'        },
  { value: 'continuous', label: 'Continuous' },
  { value: 'categorical',label: 'Categorical'},
]
const strengthOptions = [
  { value: 'all',       label: 'All'      },
  { value: 'strong',    label: 'Strong'   },
  { value: 'moderate',  label: 'Moderate+'},
  { value: 'weak',      label: 'Weak+'    },
]
const sortOptions = [
  { value: 'metric', label: 'Strength' },
  { value: 'name',   label: 'Name'     },
]

const STRENGTH_RANK = { strong: 3, moderate: 2, weak: 1, negligible: 0 }

const filtered = computed(() => {
  const q = search.value.trim().toLowerCase()
  const minRank = STRENGTH_RANK[strengthFilter.value] ?? -1

  let rows = props.rows.filter(row => {
    if (q && !row.column.toLowerCase().includes(q)) return false
    if (intentFilter.value !== 'all' && row.col_other_intent !== intentFilter.value) return false
    if (strengthFilter.value !== 'all' && (STRENGTH_RANK[row.strength] ?? 0) < minRank) return false
    return true
  })

  if (sortBy.value === 'metric') {
    rows = [...rows].sort((a, b) => Math.abs(b.metric) - Math.abs(a.metric))
  } else {
    rows = [...rows].sort((a, b) => a.column.localeCompare(b.column))
  }
  return rows
})

function formatMetric(val, type) {
  if (val == null) return '-'
  const v = Number(val)
  if (type === 'eta_squared') return `η²=${v.toFixed(3)}`
  if (type === 'cramers_v')   return `V=${v.toFixed(3)}`
  const sign = v >= 0 ? '+' : ''
  return `r=${sign}${v.toFixed(3)}`
}

function metricLabel(type) {
  const map = {
    pearson_r:       'Pearson r',
    point_biserial_r:'Point-biserial',
    cramers_v:       "Cramér's V",
    eta_squared:     'Eta squared',
  }
  return map[type] ?? type
}

function metricClass(val, type) {
  const v = Math.abs(val ?? 0)
  if (type === 'eta_squared') {
    if (v >= 0.14) return 'metric-strong'
    if (v >= 0.06) return 'metric-moderate'
    return ''
  }
  if (type === 'cramers_v') {
    if (v >= 0.5) return 'metric-strong'
    if (v >= 0.3) return 'metric-moderate'
    return ''
  }
  if (v >= 0.7) return 'metric-strong'
  if (v >= 0.4) return 'metric-moderate'
  return ''
}
</script>

<style scoped>
.assoc-table-wrap { display: flex; flex-direction: column; gap: 12px; }

.controls-row {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

/* Search */
.search-wrap { position: relative; display: flex; align-items: center; }
.search-icon { position: absolute; left: 9px; font-size: 15px; color: #64748b; pointer-events: none; }
.search-input {
  width: 180px; padding: 6px 28px 6px 28px;
  background: #1e293b; border: 1px solid #334155; border-radius: 7px;
  color: #e2e8f0; font-size: 13px; outline: none;
}
.search-input:focus { border-color: #60a5fa; }
.search-input::placeholder { color: #64748b; }
.search-clear {
  position: absolute; right: 7px; background: none; border: none;
  color: #64748b; cursor: pointer; font-size: 11px; padding: 2px;
}
.search-clear:hover { color: #e2e8f0; }

/* Chips */
.filter-chips, .sort-wrap { display: flex; align-items: center; gap: 4px; }
.chip-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.4px; white-space: nowrap; }
.chip {
  padding: 4px 10px; background: #1e293b; border: 1px solid #334155;
  border-radius: 12px; color: #94a3b8; font-size: 12px; cursor: pointer;
  transition: all 0.12s;
}
.chip:hover  { border-color: #60a5fa; color: #e2e8f0; }
.chip.active { background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }

/* Table */
.assoc-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.assoc-table th {
  text-align: left; padding: 8px 12px;
  color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px;
  border-bottom: 1px solid #334155;
}
.th-metric { width: 280px; }
.assoc-table td { padding: 9px 12px; border-bottom: 1px solid #1e293b; color: #e2e8f0; }
.assoc-row { cursor: pointer; transition: background 0.1s; }
.assoc-row:hover td  { background: #1e293b; }
.assoc-row.selected td { background: #1e3a5f; }

.td-name { font-weight: 500; color: #93c5fd; }

/* Intent badges */
.intent-badge {
  padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500;
}
.intent-continuous  { background: #1e3a5f; color: #60a5fa; }
.intent-categorical { background: #1a2e1a; color: #4ade80; }
.intent-boolean     { background: #2e1a2e; color: #c084fc; }
.intent-datetime    { background: #2e2510; color: #fb923c; }
.intent-id          { background: #1e293b; color: #64748b; }

/* Metric cell */
.metric-cell { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.metric-type-label { font-size: 10px; color: #64748b; min-width: 80px; }
.metric-val { font-size: 13px; font-weight: 700; color: #94a3b8; font-family: monospace; }
.metric-strong   { color: #4ade80; }
.metric-moderate { color: #fb923c; }



/* Strength badges */
.strength-badge {
  padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; text-transform: capitalize;
}
.str-strong    { background: #14291a; color: #4ade80; }
.str-moderate  { background: #2e1f0a; color: #fb923c; }
.str-weak      { background: #1e2a3a; color: #60a5fa; }
.str-negligible{ background: #1e293b; color: #64748b; }

.empty { color: #64748b; font-size: 13px; padding: 24px 0; text-align: center; }
.result-count { font-size: 11px; color: #64748b; text-align: right; }
</style>
