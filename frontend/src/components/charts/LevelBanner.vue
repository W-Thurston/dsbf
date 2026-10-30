<template>
  <div class="banner" :data-mark="mark" role="status">
    <TrafficDot :mark="mark" :size="14" />
    <strong class="head" data-role="headline">{{ headline }}</strong>
    <span v-if="detail" class="muted">{{ detail }}</span>
  </div>
</template>

<script setup lang="ts">
import type { Mark } from '../shell/health'
import TrafficDot from '../shell/TrafficDot.vue'

defineProps<{ mark: Mark; headline: string; detail?: string }>()
</script>

<style scoped>
.banner {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 12px;
  padding: 12px 16px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--level-none);
  border-radius: 8px;
  background: var(--surface);
}
.banner[data-mark='green'] {
  border-left-color: var(--level-green);
}
.banner[data-mark='blue'] {
  border-left-color: var(--level-blue);
}
.banner[data-mark='amber'] {
  border-left-color: var(--level-amber);
}
.banner[data-mark='red'] {
  border-left-color: var(--level-red);
}
</style>
