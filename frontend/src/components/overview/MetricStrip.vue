<template>
  <ul class="strip" aria-label="Dataset at a glance" data-role="metric-strip">
    <li v-for="metric in metrics" :key="metric.label" :data-metric="metric.label">
      <span class="value" :class="metric.tone && `tone-${metric.tone}`">{{ metric.value }}</span>
      <span class="label">{{ metric.label }}</span>
      <span v-if="metric.note" class="note">{{ metric.note }}</span>
    </li>
  </ul>
</template>

<script setup lang="ts">
import type { Tone } from './overview'

export interface Metric {
  label: string
  value: string
  note?: string
  tone?: Tone | 'green'
}

defineProps<{ metrics: Metric[] }>()
</script>

<style scoped>
.strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
  list-style: none;
  margin: 0;
  padding: 0;
}
li {
  display: flex;
  flex-direction: column;
  gap: 2px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
}
.value {
  font-size: 22px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.label {
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.4px;
}
.note {
  font-size: 12px;
  color: var(--text-muted);
}
.tone-amber {
  color: var(--level-amber);
}
.tone-red {
  color: var(--level-red);
}
.tone-green {
  color: var(--level-green);
}
</style>
