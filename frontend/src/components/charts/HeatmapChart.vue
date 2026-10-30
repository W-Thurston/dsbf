<template>
  <DsbfChart
    :option="option"
    :title="title"
    :height="height"
    :empty="empty"
    :empty-message="emptyMessage"
    :export-name="exportName"
  />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useThemeStore } from '../../stores/theme'
import DsbfChart from './DsbfChart.vue'
import { heatmap, type HeatCell } from './options'

const props = withDefaults(
  defineProps<{
    xLabels: string[]
    yLabels: string[]
    cells: HeatCell[]
    min: number
    max: number
    signed?: boolean
    title?: string
    height?: string
    emptyMessage?: string
  }>(),
  { title: '', height: '260px', emptyMessage: '' },
)

const theme = computed(() => useThemeStore().theme)
const empty = computed(() => props.cells.length === 0)
const exportName = 'dsbf-heatmap'
const option = computed(() =>
  heatmap(props.xLabels, props.yLabels, props.cells, theme.value, {
    min: props.min,
    max: props.max,
    signed: props.signed,
  }),
)
</script>
