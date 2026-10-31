<template>
  <div class="meter" :data-name="label">
    <span class="label">{{ label }}</span>
    <span
      class="track"
      role="img"
      :aria-label="`${label} strength ${value.toFixed(2)}; white noise reaches ${floor.toFixed(2)}`"
    >
      <span class="fill" :style="{ width: `${pct(value)}%` }" />
      <span class="floor" :style="{ left: `${pct(floor)}%` }" title="white-noise floor" />
    </span>
    <span class="value">{{ value.toFixed(2) }} · {{ words }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { strengthWords } from './timeseries'

const props = defineProps<{ label: string; value: number; floor: number }>()
const words = computed(() => strengthWords(props.value, props.floor))
const pct = (v: number) => Math.max(0, Math.min(v, 1)) * 100
</script>

<style scoped>
.meter {
  display: grid;
  grid-template-columns: 72px minmax(120px, 1fr) auto;
  align-items: center;
  gap: 8px;
}
.track {
  position: relative;
  height: 8px;
  border-radius: 4px;
  background: var(--border);
}
.fill {
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 4px;
  background: var(--link);
}
.floor {
  position: absolute;
  top: -3px;
  bottom: -3px;
  width: 2px;
  background: var(--text);
}
.value {
  font-size: 13px;
}
</style>
