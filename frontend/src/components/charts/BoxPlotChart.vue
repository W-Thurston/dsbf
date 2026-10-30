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
import { boxPlot, type BoxRow } from './options'

const props = withDefaults(
  defineProps<{ rows: BoxRow[]; title?: string; height?: string; emptyMessage?: string }>(),
  { title: '', height: '260px', emptyMessage: '' },
)

const theme = computed(() => useThemeStore().theme)
const empty = computed(() => props.rows.length === 0)
const exportName = 'dsbf-boxplot'
const option = computed(() => boxPlot(props.rows, theme.value))
</script>
