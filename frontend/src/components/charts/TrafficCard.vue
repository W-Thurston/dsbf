<template>
  <section class="card traffic" :data-mark="mark" :aria-label="title">
    <header class="head">
      <TrafficDot :mark="mark" :size="12" />
      <h3 class="title">{{ title }}</h3>
    </header>
    <p class="summary" data-role="summary">{{ summary }}</p>
    <slot />
  </section>
</template>

<script setup lang="ts">
import type { Mark } from '../shell/health'
import TrafficDot from '../shell/TrafficDot.vue'

defineProps<{ title: string; mark: Mark; summary: string }>()
</script>

<style scoped>
.head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.title {
  font-size: 15px;
}
.summary {
  margin: 6px 0;
  color: var(--text-muted);
}
.traffic[data-mark='none'] {
  border-style: dashed;
}
</style>
