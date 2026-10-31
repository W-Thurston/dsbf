<template>
  <ul class="dots" aria-label="Quality dimensions of the latest run">
    <li v-for="dot in dots" :key="dot.name" :data-dimension="dot.name" :data-mark="dot.mark">
      <TrafficDot :mark="dot.mark" :size="12" />
      <span class="visually-hidden">{{ dot.label }}: {{ words(dot.mark) }}</span>
    </li>
  </ul>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Mark } from '../shell/health'
import { dimensionMarks } from './dimensionMarks'
import TrafficDot from '../shell/TrafficDot.vue'
import type { components } from '../../generated/api'

const props = defineProps<{ levels: components['schemas']['DatasetSummary']['dimension_levels'] }>()

const WORDS: Record<Mark, string> = {
  green: 'all clear',
  blue: 'minor',
  amber: 'warning',
  red: 'serious',
  none: 'not assessed',
}
const words = (mark: Mark) => WORDS[mark]

const dots = computed(() => dimensionMarks(props.levels))
</script>

<style scoped>
.dots {
  display: flex;
  gap: 8px;
  list-style: none;
  margin: 0;
  padding: 0;
}
li {
  display: inline-flex;
  position: relative;
}
</style>
