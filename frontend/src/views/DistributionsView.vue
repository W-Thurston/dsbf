<template>
  <p v-if="data.error.value" class="error">{{ data.error.value }}</p>
  <p v-else-if="data.columns.value === null" class="loading">Loading…</p>
  <p v-else-if="!selected" class="muted card">This run has no columns.</p>
  <div v-else class="layout">
    <ColumnBrowser
      :run-id="runId"
      :columns="data.columns.value"
      :warned="warned"
      :selected="selected"
    />
    <ColumnDetail :run-id="runId" :column="selected" :data="data" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ColumnBrowser from '../components/column/ColumnBrowser.vue'
import ColumnDetail from '../components/column/ColumnDetail.vue'
import { warnedColumns } from '../components/column/distributions'
import { useDistributionsData } from '../composables/useDistributionsData'

// With no column in the path the first one is shown, so the tab opens on a chart.
const props = defineProps<{ runId: string; column?: string }>()

const data = useDistributionsData(() => props.runId)
const selected = computed(() => props.column ?? data.columns.value?.[0]?.name ?? '')
const warned = computed(() => warnedColumns(data.warnings.value))
</script>

<style scoped>
.layout {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
@media (max-width: 900px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
