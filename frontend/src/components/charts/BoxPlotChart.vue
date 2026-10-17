<!-- frontend/src/components/charts/BoxPlotChart.vue -->

<template>
  <DsbfChart
    :option="chartOption"
    :title="column"
    subtitle="Box Plot"
    :theme="theme"
    :empty="isEmpty"
    empty-message="No outlier data available for this column."
    :export-name="`${column}_boxplot`"
    fill-parent
  />
</template>

<script setup>
import { computed } from 'vue'
import DsbfChart from './DsbfChart.vue'

const props = defineProps({
  column: { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
  theme:  { type: String, default: 'dark' },
})

const stats    = computed(() => props.tasks?.summarize_numeric?.data?.[props.column] ?? null)
const iqr      = computed(() => props.tasks?.detect_outliers?.data?.[props.column]?.iqr ?? null)
const isEmpty  = computed(() => !stats.value || !iqr.value)

const chartOption = computed(() => {
  const s = stats.value
  const o = iqr.value
  if (!s || !o) return {}

  const q1     = o.q1            ?? s['25%']
  const q3     = o.q3            ?? s['75%']
  const median = s['50%']        ?? ((q1 + q3) / 2)
  const lf     = o.lower_fence
  const uf     = o.upper_fence
  // Whiskers: extend to the most extreme non-outlier value
  const lw     = Math.max(s.min ?? lf, lf)
  const uw     = Math.min(s.max ?? uf, uf)

  const isDark    = props.theme !== 'light'
  const boxColor  = isDark ? '#3b82f6' : '#2563eb'
  const outlColor = isDark ? '#f87171' : '#ef4444'

  return {
    animation: true,
    animationDuration: 300,

    grid: { top: 24, right: 32, bottom: 40, left: 72, containLabel: false },

    tooltip: {
      trigger: 'item',
      confine: true,
      formatter() {
        const lines = [
          `<b>${props.column}</b>`,
          `Min (fence):&nbsp; ${lw.toFixed(3)}`,
          `Q1:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ${q1.toFixed(3)}`,
          `Median:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ${median.toFixed(3)}`,
          `Q3:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ${q3.toFixed(3)}`,
          `Max (fence):&nbsp; ${uw.toFixed(3)}`,
          `IQR:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ${o.iqr_val?.toFixed(3) ?? '–'}`,
        ]
        if ((o.outlier_count ?? 0) > 0) {
          lines.push(`<span style="color:${outlColor}">Outliers: ${o.outlier_count} (${(o.outlier_pct * 100).toFixed(2)}%)</span>`)
        }
        return lines.join('<br/>')
      },
    },

    xAxis: {
      type: 'value',
      scale: true,
      axisLabel: { fontSize: 12, formatter: v => v.toFixed(2) },
      splitNumber: 4,
    },

    yAxis: {
      type: 'category',
      data: [props.column],
      axisLabel: {
        fontSize: 12,
        width: 64,
        overflow: 'truncate',
      },
      axisTick:  { show: false },
      axisLine:  { show: false },
    },

    series: [
      {
        name: props.column,
        type: 'boxplot',
        // ECharts boxplot: [min_whisker, Q1, median, Q3, max_whisker]
        data: [[lw, q1, median, q3, uw]],
        itemStyle: {
          color: `${boxColor}28`,
          borderColor: boxColor,
          borderWidth: 2,
        },
        emphasis: {
          itemStyle: { shadowBlur: 6, shadowColor: `${boxColor}66` },
        },
      },
      // Outlier marker: if outliers exist, show a single point at the
      // actual min/max to indicate the extent of the extreme values
      ...(o.outlier_count > 0 ? [{
        name: 'Outlier extent',
        type: 'scatter',
        data: [
          ...(lw > s.min ? [[s.min, props.column]] : []),
          ...(uw < s.max ? [[s.max, props.column]] : []),
        ],
        symbol: 'diamond',
        symbolSize: 8,
        itemStyle: { color: outlColor, opacity: 0.85 },
        tooltip: {
          formatter: params =>
            `<b>Outlier</b><br/>Value: ${params.value[0].toFixed(3)}<br/>` +
            `${o.outlier_count} outlier${o.outlier_count !== 1 ? 's' : ''} total`,
        },
      }] : []),
    ],
  }
})
</script>
