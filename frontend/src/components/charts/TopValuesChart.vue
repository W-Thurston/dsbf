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
import { topValues, type CountItem } from './options'

const props = withDefaults(
  defineProps<{ items: CountItem[]; title?: string; height?: string; emptyMessage?: string }>(),
  { title: '', height: '260px', emptyMessage: '' },
)

const theme = computed(() => useThemeStore().theme)
const empty = computed(() => props.items.length === 0)
const exportName = 'dsbf-top-values'
const option = computed(() => topValues(props.items, theme.value))
</script>
