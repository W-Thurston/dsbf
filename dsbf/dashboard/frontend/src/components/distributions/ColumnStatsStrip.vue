<template>
  <div class="stats-strip">
    <div
      v-for="stat in stats"
      :key="stat.label"
      class="stat-item"
      :class="stat.cls"
    >
      <span class="stat-label">
        {{ stat.label }}
        <TooltipIcon :text="stat.tooltip" direction="down" align="center" />
      </span>
      <span class="stat-value">{{ stat.value }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  column: { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
})

function fmt(v, decimals = 2) {
  if (v == null) return '—'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}
function fmtCount(v) {
  if (v == null) return '—'
  return Number(v).toLocaleString()
}
function fmtPct(v) {
  if (v == null) return '—'
  return `${(v * 100).toFixed(1)}%`
}

const stats = computed(() => {
  const col    = props.column
  const intent = props.tasks.infer_types?.data?.[col]?.analysis_intent_dtype ?? 'unknown'
  const nm     = props.tasks.summarize_numeric?.data?.[col]               ?? {}
  const skew   = props.tasks.detect_skewness?.data?.[col]                 ?? null
  const nullPct = props.tasks.summarize_nulls?.data?.null_percentages?.[col] ?? null
  const unique  = props.tasks.summarize_unique?.data?.[col]               ?? null
  const dom     = props.tasks.detect_single_dominant_value?.data?.[col]   ?? {}
  const rowCount = props.tasks.summarize_dataset_shape?.data?.num_rows    ?? null

  if (intent === 'continuous') {
    const p25 = nm['25%'] ?? null
    const p75 = nm['75%'] ?? null
    const iqr = (p25 != null && p75 != null) ? p75 - p25 : null
    const skewCls = skew == null ? '' : Math.abs(skew) > 2 ? 'stat-warn-high' : Math.abs(skew) > 1 ? 'stat-warn-mid' : ''
    const nullCls = nullPct == null ? '' : nullPct >= 0.2 ? 'stat-warn-high' : nullPct >= 0.05 ? 'stat-warn-mid' : ''
    return [
      { label: 'Count',   value: fmtCount(nm.count),  tooltip: 'Number of non-null values.',                                              cls: '' },
      { label: 'Mean',    value: fmt(nm.mean),         tooltip: 'Arithmetic mean. Sensitive to outliers — compare with median.',           cls: '' },
      { label: 'Median',  value: fmt(nm['50%']),       tooltip: 'The middle value (50th percentile). More robust to outliers than mean.',  cls: '' },
      { label: 'Std Dev', value: fmt(nm.std),          tooltip: 'Standard deviation. Measures spread around the mean.',                   cls: '' },
      { label: 'Min',     value: fmt(nm.min),          tooltip: 'Smallest observed value.',                                               cls: '' },
      { label: 'Max',     value: fmt(nm.max),          tooltip: 'Largest observed value.',                                                cls: '' },
      { label: 'Skew',    value: fmt(skew),            tooltip: 'Skewness: 0 = symmetric, >1 = right tail, <-1 = left tail. Values beyond ±2 indicate heavy skew that can affect linear models.', cls: skewCls },
      { label: 'IQR',     value: fmt(iqr),             tooltip: 'Interquartile range (p75 − p25). A robust measure of spread that ignores outliers.',  cls: '' },
      { label: 'Null %',  value: fmtPct(nullPct),      tooltip: 'Percentage of rows with a missing value.',                               cls: nullCls },
    ]
  }

  if (intent === 'categorical' || intent === 'text') {
    const topProp = dom.mode_proportion ?? null
    const topCls  = topProp != null && topProp >= 0.9 ? 'stat-warn-high' : topProp >= 0.7 ? 'stat-warn-mid' : ''
    const nullCls = nullPct == null ? '' : nullPct >= 0.2 ? 'stat-warn-high' : nullPct >= 0.05 ? 'stat-warn-mid' : ''
    const cardinalityRatio = (unique != null && rowCount != null) ? unique / rowCount : null
    const cardCls = cardinalityRatio != null && cardinalityRatio > 0.9 ? 'stat-warn-high' : ''
    return [
      { label: 'Count',     value: fmtCount(rowCount),          tooltip: 'Total number of rows in the dataset.',                                     cls: '' },
      { label: 'Unique',    value: fmtCount(unique),            tooltip: 'Number of distinct values. Very high uniqueness relative to row count may indicate an ID column.', cls: cardCls },
      { label: 'Top Value', value: dom.mode != null ? String(dom.mode).slice(0, 20) : '—', tooltip: 'The most frequently occurring value.',           cls: '' },
      { label: 'Top %',     value: fmtPct(topProp),             tooltip: 'Proportion of rows containing the most common value. Very high values (>90%) indicate a near-constant column.', cls: topCls },
      { label: 'Null %',    value: fmtPct(nullPct),             tooltip: 'Percentage of rows with a missing value.',                                  cls: nullCls },
    ]
  }

  if (intent === 'boolean') {
    const vc   = props.tasks.summarize_value_counts?.data?.[col] ?? {}
    const vals = Object.entries(vc).sort((a, b) => b[1] - a[1])
    const total = vals.reduce((s, [, n]) => s + n, 0)
    const topVal  = vals[0]?.[0] ?? '—'
    const topN    = vals[0]?.[1] ?? null
    const topProp = total > 0 && topN != null ? topN / total : null
    const imbalCls = topProp != null && topProp > 0.9 ? 'stat-warn-high' : topProp > 0.75 ? 'stat-warn-mid' : ''
    const nullCls  = nullPct == null ? '' : nullPct >= 0.2 ? 'stat-warn-high' : nullPct >= 0.05 ? 'stat-warn-mid' : ''
    return [
      { label: 'Count',      value: fmtCount(total),   tooltip: 'Total number of non-null values.',                                         cls: '' },
      { label: 'Unique',     value: fmtCount(unique),  tooltip: 'Number of distinct values (typically 2 for a boolean).',                   cls: '' },
      { label: 'Top Value',  value: String(topVal),    tooltip: 'The most frequently occurring value.',                                     cls: '' },
      { label: 'Top %',      value: fmtPct(topProp),   tooltip: 'Balance of the dominant class. >90% indicates a severely imbalanced column that may be unhelpful for classification.', cls: imbalCls },
      { label: 'Null %',     value: fmtPct(nullPct),   tooltip: 'Percentage of rows with a missing value.',                                cls: nullCls },
    ]
  }

  // Fallback for datetime / unknown
  const nullCls = nullPct == null ? '' : nullPct >= 0.2 ? 'stat-warn-high' : nullPct >= 0.05 ? 'stat-warn-mid' : ''
  return [
    { label: 'Unique', value: fmtCount(unique),  tooltip: 'Number of distinct values.', cls: '' },
    { label: 'Null %', value: fmtPct(nullPct),   tooltip: 'Percentage of rows with a missing value.', cls: nullCls },
  ]
})
</script>

<style scoped>
.stats-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 16px;
}

.stat-item {
  flex: 1 1 0;
  min-width: 80px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 12px;
  border-right: 1px solid #1e293b;
  gap: 4px;
}
.stat-item:last-child { border-right: none; }

.stat-label {
  font-size: 10px;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  display: flex;
  align-items: center;
  gap: 3px;
  white-space: nowrap;
}

.stat-value {
  font-size: 14px;
  font-weight: 600;
  color: #e2e8f0;
  font-family: monospace;
}

.stat-warn-high .stat-value { color: #f87171; }
.stat-warn-mid  .stat-value { color: #fb923c; }
</style>
