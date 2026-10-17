<!-- frontend/src/components/charts/ValueCountsChart.vue

  Horizontal bar chart of value counts for categorical/boolean columns.
  Percentages are always of the WHOLE column (num_rows), not of the subset
  shown — so "5%" means this value makes up 5% of all rows, not 5% of the
  top-N. Shows up to topN categories with a built-in scroll when there are
  more, so no data is hidden.
-->

<template>
  <DsbfChart
    :option="chartOption"
    :title="column"
    subtitle="Value Counts"
    :theme="theme"
    :empty="isEmpty"
    empty-message="No value count data available for this column."
    :export-name="`${column}_value_counts`"
    fill-parent
  />
</template>

<script setup>
import { computed } from 'vue'
import DsbfChart from './DsbfChart.vue'

const props = defineProps({
  column: { type: String, required: true },
  tasks: { type: Object, default: () => ({}) },
  theme: { type: String, default: 'dark' },
  // Maximum bars visible at once before scroll kicks in
  visibleRows: { type: Number, default: 12 },
})

const rawCounts = computed(() => props.tasks?.summarize_value_counts?.data?.[props.column] ?? null)

// True total row count from dataset shape — used for whole-column percentages
const totalRows = computed(() => props.tasks?.summarize_dataset_shape?.data?.num_rows ?? null)

const isEmpty = computed(() => !rawCounts.value || !Object.keys(rawCounts.value).length)

// All entries sorted by count descending — no artificial cap
const entries = computed(() => {
  const raw = rawCounts.value
  if (!raw) return []
  return Object.entries(raw)
    .filter(([, v]) => typeof v === 'number')
    .sort((a, b) => b[1] - a[1])
})

// Sum of all counts shown (may be less than totalRows due to nulls)
const countTotal = computed(() => entries.value.reduce((s, [, v]) => s + v, 0))

// Use totalRows for percentage denominator when available,
// fall back to the sum of all counts in the value_counts dict
const pctDenominator = computed(() => totalRows.value ?? countTotal.value)

// Chart height: 32px per visible row + 60px overhead, capped
// (kept for reference — chart now fills parent via fill-parent prop)

const chartOption = computed(() => {
  const items = entries.value
  if (!items.length) return {}

  const n = pctDenominator.value
  const isDark = props.theme !== 'light'
  const maxCount = items[0]?.[1] ?? 1

  function barColor(count) {
    const pct = count / maxCount // relative to max for coloring
    if (pct >= 0.9) return isDark ? '#60a5fa' : '#2563eb' // dominant
    if (pct >= 0.5) return isDark ? '#3b82f6' : '#3b82f6'
    return isDark ? '#1d4ed8' : '#93c5fd' // minor
  }

  // Reversed so highest bar is at top
  const reversed = [...items].reverse()

  // How many rows are off-screen (need scrolling)?
  const needsScroll = items.length > props.visibleRows
  // Show bottom N rows by default (most common = top = end of reversed array)
  const scrollStart = needsScroll ? Math.max(0, 100 - (props.visibleRows / items.length) * 100) : 0

  return {
    animation: true,
    animationDuration: 350,

    grid: {
      top: 8,
      right: 72, // room for percentage labels
      bottom: needsScroll ? 28 : 16,
      left: 16,
      containLabel: true,
    },

    tooltip: {
      trigger: 'axis',
      confine: true,
      axisPointer: { type: 'shadow' },
      formatter(params) {
        const count = params[0].value
        const colPct = n > 0 ? ((count / n) * 100).toFixed(1) : '–'
        return [
          `<b>${params[0].name}</b>`,
          `Count: <b>${count.toLocaleString()}</b>`,
          `${colPct}% of column`,
        ].join('<br/>')
      },
    },

    xAxis: {
      type: 'value',
      axisLabel: {
        fontSize: 12,
        formatter: (v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v),
      },
      splitNumber: 3,
    },

    yAxis: {
      type: 'category',
      data: reversed.map(([k]) => String(k)),
      axisLabel: {
        fontSize: 12,
        width: 110,
        overflow: 'truncate',
      },
    },

    series: [
      {
        type: 'bar',
        data: reversed.map(([, v]) => ({
          value: v,
          itemStyle: {
            color: barColor(v),
            borderRadius: [0, 3, 3, 0],
          },
        })),
        barMaxWidth: 40,
        label: {
          show: true,
          position: 'right',
          fontSize: 11,
          color: isDark ? '#94a3b8' : '#64748b',
          // Show percentage of WHOLE column
          formatter: (params) => (n > 0 ? `${((params.value / n) * 100).toFixed(1)}%` : ''),
        },
      },
    ],

    // Scrollable y-axis when there are more categories than visibleRows
    ...(needsScroll
      ? {
          dataZoom: [
            {
              type: 'inside',
              yAxisIndex: 0,
              start: scrollStart,
              end: 100,
              zoomOnMouseWheel: false,
              moveOnMouseMove: true,
              moveOnMouseWheel: true,
            },
            {
              type: 'slider',
              yAxisIndex: 0,
              width: 12,
              right: 0,
              start: scrollStart,
              end: 100,
              showDetail: false,
              showDataShadow: false,
              borderColor: 'transparent',
              backgroundColor: isDark ? '#1e293b' : '#f1f5f9',
              fillerColor: isDark ? '#334155' : '#cbd5e1',
              handleStyle: {
                color: isDark ? '#60a5fa' : '#2563eb',
                borderColor: 'transparent',
              },
            },
          ],
        }
      : {}),
  }
})
</script>
