<template>
  <section class="health card" aria-label="Data health" data-role="health-bar">
    <p v-if="!dimensions" class="muted" data-role="health-missing">
      This run has no quality summary, so no dimension is assessed.
    </p>
    <ul class="dims">
      <li
        v-for="d in bar"
        :key="d.name"
        :class="['dim', { dimmed: !d.relevant }]"
        :data-dimension="d.name"
        :data-mark="d.mark"
        :title="d.detail"
        tabindex="0"
      >
        <TrafficDot :mark="d.mark" :size="12" />
        <span class="label">{{ d.label }}</span>
        <span class="summary" data-role="summary">{{ d.summary }}</span>
        <span class="tip" role="tooltip">{{ d.detail }}</span>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { QualityOutput } from '../quality/quality'
import TrafficDot from './TrafficDot.vue'
import { healthBar, type Tab } from './health'

const props = defineProps<{ dimensions: QualityOutput | null; tab: Tab }>()

const bar = computed(() =>
  healthBar(props.dimensions ?? { total_columns: 0, dimensions: {}, clean_columns: [] }, props.tab),
)
</script>

<style scoped>
.health {
  padding: 10px 16px;
}
.dims {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 28px;
  list-style: none;
  margin: 0;
  padding: 0;
}
.dim {
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  outline-offset: 4px;
}
.dimmed {
  opacity: 0.45;
}
.label {
  font-weight: 600;
}
.summary {
  color: var(--text-muted);
  font-size: 13px;
}
.tip {
  display: none;
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 10;
  margin-top: 8px;
  width: 280px;
  padding: 8px 10px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 12px;
  color: var(--text);
  box-shadow: 0 4px 12px rgb(0 0 0 / 0.3);
}
.dim:hover .tip,
.dim:focus .tip {
  display: block;
}
</style>
