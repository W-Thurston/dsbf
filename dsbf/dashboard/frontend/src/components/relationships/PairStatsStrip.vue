<template>
  <div class="pair-stats-strip card" v-if="stats.length">
    <div class="stat" v-for="s in stats" :key="s.label">
      <span class="stat-label">
        {{ s.label }}
        <TooltipIcon v-if="s.tooltip" :text="s.tooltip" />
      </span>
      <span class="stat-value" :class="s.cls">{{ s.value }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  colA:       { type: String,  default: '' },
  colB:       { type: String,  default: '' },
  intentA:    { type: String,  default: 'continuous' },
  intentB:    { type: String,  default: 'continuous' },
  metric:     { type: Number,  default: null },
  metricType: { type: String,  default: '' },
  strength:   { type: String,  default: '' },
  sampleSize: { type: Number,  default: null },
})

const METRIC_TOOLTIPS = {
  pearson_r:        'Pearson r measures linear correlation between two continuous variables. Range: -1 to +1.',
  point_biserial_r: 'Point-biserial r measures the correlation between a continuous variable and a binary variable. Range: -1 to +1.',
  cramers_v:        "Cramér's V measures association between two categorical variables. Range: 0 (no association) to 1 (perfect association).",
  eta_squared:      'Eta squared (η²) is the proportion of variance in the continuous variable explained by the categorical variable. Range: 0 to 1.',
}

const METRIC_LABELS = {
  pearson_r:        'Pearson r',
  point_biserial_r: 'Point-biserial r',
  cramers_v:        "Cramér's V",
  eta_squared:      'Eta squared (η²)',
}

const STRENGTH_CLS = {
  strong:     'val-strong',
  moderate:   'val-moderate',
  weak:       'val-weak',
  negligible: '',
}

function metricDisplay(val, type) {
  if (val == null) return '-'
  const v = Number(val)
  if (type === 'eta_squared') return v.toFixed(4)
  if (type === 'cramers_v')   return v.toFixed(4)
  const sign = v >= 0 ? '+' : ''
  return `${sign}${v.toFixed(4)}`
}

const stats = computed(() => {
  if (props.metric == null) return []
  const items = [
    {
      label:   METRIC_LABELS[props.metricType] ?? props.metricType,
      value:   metricDisplay(props.metric, props.metricType),
      tooltip: METRIC_TOOLTIPS[props.metricType] ?? null,
      cls:     STRENGTH_CLS[props.strength] ?? '',
    },
    {
      label: 'Strength',
      value: props.strength ? props.strength.charAt(0).toUpperCase() + props.strength.slice(1) : '-',
      cls:   STRENGTH_CLS[props.strength] ?? '',
    },
    {
      label:   'Column A Intent',
      value:   props.intentA,
    },
    {
      label:   'Column B Intent',
      value:   props.intentB,
    },
  ]

  if (props.sampleSize) {
    items.push({
      label:   'Sample Size (n)',
      value:   props.sampleSize.toLocaleString(),
      tooltip: 'Number of complete (non-null) row pairs used to compute the metric.',
    })
  }

  return items
})
</script>

<style scoped>
.pair-stats-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  padding: 14px 20px;
}
.stat {
  display: flex;
  flex-direction: column;
  gap: 3px;
  flex: 1;
  min-width: 120px;
  padding: 6px 16px;
  border-right: 1px solid #1e293b;
}
.stat:last-child { border-right: none; }
.stat-label {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  display: flex;
  align-items: center;
  gap: 4px;
}
.stat-value {
  font-size: 18px;
  font-weight: 700;
  color: #e2e8f0;
}
.val-strong   { color: #4ade80; }
.val-moderate { color: #fb923c; }
.val-weak     { color: #60a5fa; }
</style>
