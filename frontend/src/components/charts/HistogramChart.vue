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
import { histogramBars, type ReferenceLine } from './options'

const props = withDefaults(
  defineProps<{
    edges: number[]
    counts: number[]
    references?: ReferenceLine[]
    title?: string
    height?: string
    emptyMessage?: string
  }>(),
  { references: () => [], title: '', height: '260px', emptyMessage: '' },
)

const theme = computed(() => useThemeStore().theme)
const empty = computed(() => props.counts.length === 0)
const exportName = 'dsbf-histogram'
const option = computed(() =>
  histogramBars(props.edges, props.counts, theme.value, props.references),
)
</script>
