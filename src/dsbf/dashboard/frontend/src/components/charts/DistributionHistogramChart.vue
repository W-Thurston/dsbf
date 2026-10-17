<!-- dsbf/dashboard/frontend/src/components/charts/DistributionHistogramChart.vue -->

<template>
  <DsbfChart
    :option="chartOption"
    :title="column"
    subtitle="Distribution"
    :theme="theme"
    :empty="isEmpty"
    empty-message="No numeric summary available for this column."
    :export-name="`${column}_histogram`"
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

const stats = computed(() =>
  props.tasks?.summarize_numeric?.data?.[props.column] ?? null
)
const outlierData = computed(() =>
  props.tasks?.detect_outliers?.data?.[props.column] ?? null
)
const skewness = computed(() =>
  props.tasks?.detect_skewness?.data?.[props.column] ?? null
)
const isEmpty = computed(() => !stats.value)

// ── Build histogram bins from percentile markers ──────────────────────────────
const histBins = computed(() => {
  const s = stats.value
  if (!s) return []

  const markers = [
    { pct: 0,    val: s.min    },
    { pct: 0.01, val: s['1%']  },
    { pct: 0.05, val: s['5%']  },
    { pct: 0.25, val: s['25%'] },
    { pct: 0.50, val: s['50%'] },
    { pct: 0.75, val: s['75%'] },
    { pct: 0.95, val: s['95%'] },
    { pct: 0.99, val: s['99%'] },
    { pct: 1.00, val: s.max    },
  ].filter(m => m.val != null)

  // Deduplicate identical boundary values
  const unique = []
  for (const m of markers) {
    if (!unique.length || m.val !== unique[unique.length - 1].val) {
      unique.push(m)
    }
  }
  if (unique.length < 2) return []

  const n = s.count ?? 1
  const bins = []
  for (let i = 0; i < unique.length - 1; i++) {
    const lo  = unique[i]
    const hi  = unique[i + 1]
    const freq = (hi.pct - lo.pct) * n
    bins.push({
      lo:    lo.val,
      hi:    hi.val,
      mid:   (lo.val + hi.val) / 2,
      freq:  Math.round(freq),
      label: `${lo.val.toFixed(2)} – ${hi.val.toFixed(2)}`,
    })
  }
  return bins
})

// ── IQR fence thresholds for outlier bar coloring ─────────────────────────────
const lowerFence = computed(() => outlierData.value?.iqr?.lower_fence ?? -Infinity)
const upperFence = computed(() => outlierData.value?.iqr?.upper_fence ??  Infinity)

// ── ECharts option ────────────────────────────────────────────────────────────
const chartOption = computed(() => {
  const bins = histBins.value
  const s    = stats.value
  if (!bins.length || !s) return {}

  const isDark    = props.theme !== 'light'
  const barNormal = isDark ? '#3b82f6'              : '#2563eb'
  const barOutlier = isDark ? 'rgba(248,113,113,0.7)' : '#ef4444'
  const skew      = skewness.value

  // ── Mark lines: mean + median ───────────────────────────────────────────
  const markLineData = []
  if (s.mean != null) {
    markLineData.push({
      xAxis: s.mean,
      lineStyle: { color: '#60a5fa', type: 'dashed', width: 2 },
      label: {
        formatter: `Mean ${s.mean.toFixed(2)}`,
        position: 'insideStartTop',
        fontSize: 11,
        color: '#60a5fa',
        distance: 4,
      },
    })
  }
  if (s['50%'] != null) {
    markLineData.push({
      xAxis: s['50%'],
      lineStyle: { color: '#4ade80', type: 'solid', width: 2 },
      label: {
        formatter: `Median ${s['50%'].toFixed(2)}`,
        position: 'insideStartBottom',
        fontSize: 11,
        color: '#4ade80',
        distance: 4,
      },
    })
  }

  const skewText = skew != null
    ? `  skew ${skew > 0 ? '+' : ''}${skew.toFixed(2)}`
    : ''

  return {
    animation: true,
    animationDuration: 350,

    grid: { top: 28, right: 20, bottom: 56, left: 64 },

    tooltip: {
      trigger: 'axis',
      confine: true,
      axisPointer: { type: 'shadow' },
      formatter(params) {
        const idx = params[0]?.dataIndex
        const bin = bins[idx]
        if (!bin) return ''
        const pct = s.count > 0 ? ((bin.freq / s.count) * 100).toFixed(1) : '–'
        return [
          `<b>${bin.label}</b>`,
          `Count: <b>${bin.freq.toLocaleString()}</b>`,
          `${pct}% of column`,
        ].join('<br/>')
      },
    },

    xAxis: {
      type: 'category',
      data: bins.map(b => b.lo),
      axisLabel: {
        fontSize: 12,
        formatter: v => parseFloat(v).toFixed(Number.isInteger(parseFloat(v)) ? 0 : 2),
        rotate: bins.length > 7 ? 30 : 0,
        interval: Math.max(0, Math.floor(bins.length / 6) - 1),
        margin: 10,
      },
      axisTick: { alignWithLabel: true },
      name: `${props.column}${skewText}`,
      nameLocation: 'middle',
      nameGap: 40,
      nameTextStyle: { fontSize: 12 },
    },

    yAxis: {
      type: 'value',
      name: 'Count',
      nameTextStyle: { fontSize: 12, padding: [0, 0, 0, -48] },
      axisLabel: {
        fontSize: 12,
        formatter: v => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v,
      },
      splitNumber: 4,
    },

    series: [{
      type: 'bar',
      data: bins.map(b => ({
        value: b.freq,
        itemStyle: {
          // Bars outside IQR fences get a red tint instead of blue
          color: (isFinite(lowerFence.value) && b.lo < lowerFence.value) ||
                 (isFinite(upperFence.value) && b.hi > upperFence.value)
            ? barOutlier
            : barNormal,
          borderRadius: [2, 2, 0, 0],
        },
      })),
      // barWidth 100% makes bars truly edge-to-edge (histogram convention).
      // barCategoryGap alone is not enough on category axes in ECharts.
      barWidth: '100%',
      barCategoryGap: '0%',
      barGap: '0%',
      markLine: markLineData.length ? {
        silent: true,
        symbol: 'none',
        data: markLineData,
        animation: false,
      } : undefined,
      // Outlier fence shading removed — it rendered as a distracting wide
      // grey block. The red bar coloring already communicates the same signal.
    }],

    // Slim inside-only zoom — no external slider
    dataZoom: [{
      type: 'inside',
      xAxisIndex: 0,
      start: 0,
      end: 100,
      zoomOnMouseWheel: true,
      moveOnMouseMove: true,
    }],
  }
})
</script>
