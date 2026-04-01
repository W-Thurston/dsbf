<template>
  <div class="card run-date-card">
    <div class="card-title">
      Run Information
      <TooltipIcon
        text="Details about when and how this profiling run was executed."
        align="right"
      />
    </div>
    <div v-if="!run" class="es-not-run">Run metadata not available.</div>
    <div v-else class="run-date-grid">
      <div class="run-date-item" v-for="item in items" :key="item.label">
        <span class="run-date-label">
          {{ item.label }}
          <TooltipIcon :text="item.tooltip" align="center" />
        </span>
        <span class="run-date-value">{{ item.value }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'
import { formatDateFull } from '../../utils.js'

const props = defineProps({
  run: { type: Object, default: null },
})

const items = computed(() => {
  if (!props.run) return []
  return [
    {
      label:   'Run Date',
      tooltip: 'The date and time this profiling run was executed.',
      value:   formatDateFull(props.run.ran_at),
    },
    {
      label:   'Run Key',
      tooltip: 'Unique identifier for this run, derived from its execution timestamp.',
      value:   props.run.run_key ?? '-',
    },
  ]
})
</script>

<style scoped>
.run-date-card {
  display: flex;
  flex-direction: column;
}

.es-not-run {
  color: #475569;
  font-size: 13px;
  padding: 20px 0;
  text-align: center;
}

.run-date-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.run-date-item {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid #0f172a;
}
.run-date-item:last-child { border-bottom: none; padding-bottom: 0; }

.run-date-label {
  font-size: 12px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  display: flex;
  align-items: center;
  gap: 2px;
  white-space: nowrap;
  flex-shrink: 0;
}

.run-date-value {
  font-size: 13px;
  color: #e2e8f0;
  font-family: monospace;
  text-align: right;
  word-break: break-all;
}
</style>
