<template>
  <svg
    v-if="bars.length"
    class="sketch"
    :width="width"
    :height="height"
    role="img"
    :aria-label="label"
    data-role="row-sketch"
  >
    <rect
      v-for="(bar, i) in bars"
      :key="i"
      :x="i * step"
      :y="height - Math.max(bar.height * height, bar.height > 0 ? 1 : 0)"
      :width="Math.max(step - 1, 1)"
      :height="Math.max(bar.height * height, bar.height > 0 ? 1 : 0)"
    >
      <title>{{ bar.label }}</title>
    </rect>
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Bar } from './overview'

// A row's chart is plain SVG: a table of hundreds of columns would otherwise hold
// hundreds of canvases.
const props = withDefaults(
  defineProps<{ bars: Bar[]; label: string; width?: number; height?: number }>(),
  {
    width: 72,
    height: 22,
  },
)

const step = computed(() => props.width / Math.max(props.bars.length, 1))
</script>

<style scoped>
.sketch {
  display: block;
  fill: var(--chart-primary);
}
</style>
