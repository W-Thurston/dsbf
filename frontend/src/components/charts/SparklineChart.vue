<template>
  <svg
    class="spark"
    :viewBox="`0 0 ${width} ${height}`"
    :width="width"
    :height="height"
    role="img"
    :aria-label="label"
    data-role="sparkline"
  >
    <polyline
      v-if="points"
      :points="points"
      fill="none"
      stroke="var(--chart-primary)"
      stroke-width="1.5"
      stroke-linejoin="round"
    />
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{ values: number[]; width?: number; height?: number; label?: string }>(),
  { width: 80, height: 20, label: 'trend' },
)

// Two or more finite values make a line; a constant series is drawn flat in the middle.
const points = computed(() => {
  const values = props.values.filter(Number.isFinite)
  if (values.length < 2) return ''
  const low = Math.min(...values)
  const span = Math.max(...values) - low
  const pad = 2
  return values
    .map((v, i) => {
      const x = pad + (i / (values.length - 1)) * (props.width - 2 * pad)
      const y =
        span === 0
          ? props.height / 2
          : props.height - pad - ((v - low) / span) * (props.height - 2 * pad)
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
})
</script>

<style scoped>
.spark {
  vertical-align: middle;
}
</style>
