<template>
  <div class="strength" role="img" :aria-label="summary" data-role="strength-bar">
    <span
      v-for="s in shown"
      :key="s.label"
      class="seg"
      :style="{ flexGrow: s.value, background: `var(--level-${s.mark})` }"
      :title="`${s.label}: ${s.value}`"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Mark } from '../shell/health'

export interface Segment {
  label: string
  value: number
  mark: Mark
}

const props = defineProps<{ segments: Segment[] }>()

// A segment with nothing in it takes no room, so the bar is as wide as its total.
const shown = computed(() => props.segments.filter((s) => s.value > 0))
const summary = computed(() => props.segments.map((s) => `${s.label} ${s.value}`).join(', '))
</script>

<style scoped>
.strength {
  display: flex;
  height: 10px;
  min-width: 80px;
  border-radius: 5px;
  overflow: hidden;
  background: var(--border);
}
.seg {
  flex-basis: 0;
}
.seg + .seg {
  border-left: 1px solid var(--surface);
}
</style>
