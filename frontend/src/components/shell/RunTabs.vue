<template>
  <nav class="tabs" aria-label="Run pages">
    <router-link
      v-for="t in TABS"
      :key="t.tab"
      :to="`/runs/${runId}${t.path}`"
      :class="['tab', { active: t.tab === tab }]"
      :aria-current="t.tab === tab ? 'page' : undefined"
      :data-tab="t.tab"
      >{{ t.label }}</router-link
    >
  </nav>
</template>

<script setup lang="ts">
import { TABS, type Tab } from './health'

defineProps<{ runId: string; tab: Tab | null }>()
</script>

<style scoped>
.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  border-bottom: 1px solid var(--border);
}
.tab {
  padding: 8px 16px;
  color: var(--text-muted);
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  font-weight: 500;
}
.tab:hover {
  color: var(--text);
}
.tab.active {
  color: var(--link);
  border-bottom-color: var(--link);
}
</style>
