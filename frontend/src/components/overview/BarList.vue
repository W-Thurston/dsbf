<template>
  <ul class="bars" :aria-label="label">
    <li v-for="item in items" :key="item.label" :data-bar="item.label">
      <span class="name" :title="item.label">{{ item.label }}</span>
      <span class="track"
        ><span class="fill" :style="{ width: `${(item.count / tallest) * 100}%` }"
      /></span>
      <span class="count">{{ item.text ?? formatCount(item.count) }}</span>
    </li>
  </ul>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { formatCount } from '../../format'

export interface BarItem {
  label: string
  count: number
  text?: string
}

// Horizontal bars in plain CSS: the Overview opens on every run, and the chart library
// is a 225 KB (compressed) chunk that it should not have to fetch for a short bar list.
const props = defineProps<{ items: BarItem[]; label: string }>()
const tallest = computed(() => Math.max(...props.items.map((i) => i.count), 1))
</script>

<style scoped>
.bars {
  list-style: none;
  margin: 0;
  padding: 0;
}
li {
  display: grid;
  grid-template-columns: 130px 1fr 56px;
  gap: 10px;
  align-items: center;
  padding: 3px 0;
}
.name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.track {
  height: 12px;
  border-radius: 6px;
  background: var(--border);
  overflow: hidden;
}
.fill {
  display: block;
  height: 100%;
  background: var(--chart-primary);
}
.count {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
</style>
