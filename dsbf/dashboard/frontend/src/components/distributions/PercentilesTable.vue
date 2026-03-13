<template>
  <div class="card context-card">
    <div class="card-title">Percentile Breakdown</div>
    <table class="pct-table">
      <thead>
        <tr>
          <th v-for="p in percentiles" :key="p.label">{{ p.label }}</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td v-for="p in percentiles" :key="p.label" :class="p.cls">{{ p.value }}</td>
        </tr>
      </tbody>
    </table>

    <div v-if="outlierNote" class="outlier-note">
      <span class="outlier-icon">📌</span>
      {{ outlierNote }}
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  column: { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
})

function fmt(v) {
  if (v == null) return '-'
  const n = Number(v)
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
  return n.toPrecision(4).replace(/\.?0+$/, '')
}

const nm = computed(() => props.tasks.summarize_numeric?.data?.[props.column] ?? {})

const percentiles = computed(() => {
  const d = nm.value
  return [
    { label: 'Min',  value: fmt(d.min),    cls: '' },
    { label: 'p1',   value: fmt(d['1%']),  cls: '' },
    { label: 'p5',   value: fmt(d['5%']),  cls: '' },
    { label: 'p25',  value: fmt(d['25%']), cls: '' },
    { label: 'p50',  value: fmt(d['50%']), cls: 'pct-median' },
    { label: 'p75',  value: fmt(d['75%']), cls: '' },
    { label: 'p95',  value: fmt(d['95%']), cls: '' },
    { label: 'p99',  value: fmt(d['99%']), cls: '' },
    { label: 'Max',  value: fmt(d.max),    cls: '' },
  ]
})

const outlierNote = computed(() => {
  const d   = nm.value
  const p25 = d['25%'], p75 = d['75%'], p99 = d['99%']
  const max = d.max
  if (p25 == null || p75 == null || p99 == null || max == null) return null
  const iqr     = p75 - p25
  const upperFence = p75 + 3 * iqr
  if (max > upperFence) {
    return `Max value (${fmt(max)}) exceeds the 3×IQR upper fence (${fmt(upperFence)}), indicating potential outliers beyond p99.`
  }
  return null
})
</script>

<style scoped>
.context-card { overflow-x: auto; }

.pct-table {
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
  font-size: 12px;
}

.pct-table th {
  padding: 6px 10px;
  text-align: center;
  font-size: 10px;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  border-bottom: 1px solid #1e293b;
  white-space: nowrap;
}

.pct-table td {
  padding: 8px 10px;
  text-align: center;
  font-family: monospace;
  color: #94a3b8;
  white-space: nowrap;
}

.pct-median { color: #60a5fa !important; font-weight: 600; }

.outlier-note {
  margin-top: 10px;
  padding: 8px 10px;
  background: #3d2510;
  border: 1px solid #fb923c;
  border-radius: 6px;
  font-size: 12px;
  color: #fed7aa;
  display: flex;
  gap: 6px;
  align-items: flex-start;
  line-height: 1.4;
}
.outlier-icon { flex-shrink: 0; }
</style>
