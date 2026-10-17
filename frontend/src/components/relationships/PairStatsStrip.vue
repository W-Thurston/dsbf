<template>
  <div class="pair-stats-strip card" v-if="stats.length">
    <div class="stat" v-for="s in stats" :key="s.label">
      <span class="stat-label">
        {{ s.label }}
        <TooltipIcon v-if="s.tooltip" :text="s.tooltip" />
      </span>
      <span class="stat-value" :class="s.cls">{{ s.value }}</span>
    </div>
    <div v-if="interpretationText" class="interp-row">
      <span class="interp-text">{{ interpretationText }}</span>
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

const interpretationText = computed(() => {
  if (props.metric == null || !props.metricType || !props.strength) return null
  const v   = Number(props.metric)
  const a   = props.colA
  const b   = props.colB
  const dir = v >= 0 ? 'positive' : 'negative'
  const abs = (Math.abs(v) * 100).toFixed(0)

  if (props.metricType === 'pearson_r') {
    if (props.strength === 'negligible') return `No meaningful linear relationship between ${a} and ${b}.`
    if (props.strength === 'weak')       return `A weak ${dir} linear relationship - ${a} and ${b} move together slightly but with considerable scatter.`
    if (props.strength === 'moderate')   return `A moderate ${dir} linear relationship - as ${a} ${v >= 0 ? 'increases' : 'decreases'}, ${b} tends to follow, though with notable variation.`
    if (props.strength === 'strong')     return `A strong ${dir} linear relationship - ${a} and ${b} move closely together. Values of one are substantially predictive of the other.`
  }

  if (props.metricType === 'eta_squared') {
    if (props.strength === 'negligible') return `The groupings in ${b} explain very little of the variance in ${a}.`
    if (props.strength === 'weak')       return `The groupings in ${b} explain ${abs}% of the variance in ${a} - a small but present effect.`
    if (props.strength === 'moderate')   return `The groupings in ${b} explain ${abs}% of the variance in ${a} - a meaningful association.`
    if (props.strength === 'strong')     return `The groupings in ${b} explain ${abs}% of the variance in ${a}. The distribution of ${a} differs considerably across categories.`
  }

  if (props.metricType === 'cramers_v') {
    if (props.strength === 'negligible') return `No meaningful association between the categories of ${a} and ${b}.`
    if (props.strength === 'weak')       return `A weak association between the categories of ${a} and ${b} - knowing one tells you little about the other.`
    if (props.strength === 'moderate')   return `A moderate association between the categories of ${a} and ${b} - the distribution of one shifts meaningfully across values of the other.`
    if (props.strength === 'strong')     return `A strong association between the categories of ${a} and ${b} - the two are closely linked.`
  }

  if (props.metricType === 'point_biserial_r') {
    if (props.strength === 'negligible') return `The binary grouping in ${b} shows no meaningful difference in ${a} values between groups.`
    if (props.strength === 'weak')       return `A slight difference in ${a} between the two groups defined by ${b}.`
    if (props.strength === 'moderate')   return `A moderate difference in ${a} between the two groups defined by ${b} - the groups are meaningfully distinguishable.`
    if (props.strength === 'strong')     return `A strong difference in ${a} between the two groups defined by ${b} - the groups are clearly separated.`
  }

  return null
})
</script>

<style scoped>
.pair-stats-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  padding: 14px 20px;
  overflow: visible;
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

.interp-row {
  width: 100%;
  flex-basis: 100%;
  padding: 10px 16px 4px;
  border-top: 1px solid #1e293b;
  margin-top: 6px;
}
.interp-text {
  font-size: 13px;
  color: #94a3b8;
  line-height: 1.6;
}
</style>
