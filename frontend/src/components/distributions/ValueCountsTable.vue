<template>
  <div class="card context-card">
    <div class="card-title">Value Counts</div>

    <div v-if="!rows.length" class="no-data">No value count data available.</div>

    <div v-else class="vc-wrap">
      <table class="vc-table">
        <thead>
          <tr>
            <th class="th-val">Value</th>
            <th class="th-num">Count</th>
            <th class="th-num">% of column</th>
            <th class="th-bar" title="Bar width shows frequency relative to the most common value in this column">Relative Freq</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.value">
            <td class="val-cell" :title="row.value">{{ trunc(row.value, 28) }}</td>
            <td class="num-cell">{{ row.count.toLocaleString() }}</td>
            <td class="num-cell">{{ row.pct }}</td>
            <td class="bar-cell">
              <div class="bar-track">
                <div class="bar-fill" :style="{ width: row.barWidth }" />
              </div>
            </td>
          </tr>
        </tbody>
      </table>

      <div v-if="truncated" class="truncated-note">
        Showing top {{ maxRows }} of {{ totalUnique?.toLocaleString() }} values.
      </div>

      <div v-if="dominanceNote" class="dominant-note">
        <span class="dominant-icon">ℹ</span>
        {{ dominanceNote }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { trunc } from '../../utils.js'

const props = defineProps({
  column:  { type: String, required: true },
  tasks:   { type: Object, default: () => ({}) },
  maxRows: { type: Number, default: 20 },
})

const totalUnique = computed(() => props.tasks.summarize_unique?.data?.[props.column] ?? null)

// True total rows for whole-column percentage.
// summarize_value_counts only stores top_k values, so summing its entries
// would undercount. Use num_rows from dataset shape instead.
const numRows = computed(() =>
  props.tasks.summarize_dataset_shape?.data?.num_rows ?? null
)

const dominanceNote = computed(() => {
  const dom = props.tasks.detect_single_dominant_value?.data?.[props.column]
  if (!dom) return null
  const level = dom.dominance_level
  const pct   = dom.mode_proportion != null ? (dom.mode_proportion * 100).toFixed(1) : null
  const mode  = dom.mode != null ? String(dom.mode) : null
  if (level === 'high' && pct && mode) {
    return `"${mode}" accounts for ${pct}% of all values. This column is near-constant and may provide limited signal for analysis or modelling.`
  }
  if (level === 'medium' && pct && mode) {
    return `"${mode}" is the dominant value at ${pct}% of rows. Consider whether this reflects genuine data or a data collection artefact.`
  }
  return null
})

const rows = computed(() => {
  const vc = props.tasks.summarize_value_counts?.data?.[props.column]
  if (!vc || typeof vc !== 'object') return []

  const entries = Object.entries(vc)
    .filter(([, v]) => typeof v === 'number')
    .sort((a, b) => b[1] - a[1])

  // Use true row count for whole-column percentages; fall back to entry sum
  const denom   = numRows.value ?? entries.reduce((s, [, n]) => s + n, 0)
  const topN    = entries.slice(0, props.maxRows)
  const maxCount = topN[0]?.[1] ?? 1

  return topN.map(([val, count]) => ({
    value:    String(val),
    count,
    pct:      denom > 0 ? `${((count / denom) * 100).toFixed(1)}%` : '-',
    barWidth: `${((count / maxCount) * 100).toFixed(1)}%`,
  }))
})

const truncated = computed(() =>
  totalUnique.value != null && totalUnique.value > props.maxRows
)
</script>

<style scoped>
.context-card { overflow: hidden; }
.no-data { color: #64748b; font-size: 13px; padding: 12px 0; }

.vc-wrap { overflow-x: auto; }

.vc-table {
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
  font-size: 12px;
}

.vc-table th {
  padding: 6px 10px;
  font-size: 10px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  border-bottom: 1px solid #1e293b;
  white-space: nowrap;
}
.th-val { text-align: left; }
.th-num { text-align: right; min-width: 60px; }
.th-bar { text-align: left; width: 120px; color: #334155; font-size: 9px; }

.vc-table td {
  padding: 6px 10px;
  border-bottom: 1px solid #0f172a;
  color: #94a3b8;
  white-space: nowrap;
}
.vc-table tr:hover td { background: #263548; }

.val-cell {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: monospace;
  color: #e2e8f0 !important;
}
.num-cell { color: #94a3b8; text-align: right; min-width: 60px; }

.bar-cell { width: 120px; padding-right: 12px; }
.bar-track {
  height: 6px;
  background: #1e293b;
  border-radius: 3px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: #3b82f6;
  border-radius: 3px;
  transition: width 0.3s;
}

.truncated-note {
  margin-top: 8px;
  font-size: 11px;
  color: #64748b;
  text-align: right;
}

.dominant-note {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-top: 10px;
  padding: 8px 12px;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 6px;
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.5;
}
.dominant-icon { flex-shrink: 0; color: #60a5fa; font-size: 13px; }
</style>
