<template>
  <section class="card" aria-labelledby="pair-title">
    <h2 id="pair-title" class="card-title">Pair plot</h2>
    <form class="filters" aria-label="Choose a pair" @submit.prevent>
      <label>
        First column
        <select
          :value="first"
          @change="$emit('select', ($event.target as HTMLSelectElement).value, second)"
        >
          <option value="">choose…</option>
          <option v-for="c in columns" :key="c.name" :value="c.name">{{ c.name }}</option>
        </select>
      </label>
      <label>
        Second column
        <select
          :value="second"
          @change="$emit('select', first, ($event.target as HTMLSelectElement).value)"
        >
          <option value="">choose…</option>
          <option v-for="c in columns" :key="c.name" :value="c.name">{{ c.name }}</option>
        </select>
      </label>
    </form>
    <p v-if="!first || !second || first === second" class="muted">
      Choose two different columns to plot them from the data snapshot.
    </p>
    <p v-else-if="error" class="error" role="alert">{{ error }}</p>
    <p v-else-if="!points" class="loading">Loading…</p>
    <template v-else>
      <p class="muted" role="status">
        {{ formatCount(points.values.length) }} points<template
          v-if="points.rows_with_both > points.values.length"
        >
          (every {{ Math.ceil(points.rows_with_both / points.values.length) }}th of
          {{ formatCount(points.rows_with_both) }} rows with both)</template
        >
        from a snapshot of {{ formatCount(points.snapshot_rows) }} rows<template
          v-if="points.sampling"
          >, a {{ points.sampling.strategy }} sample of
          {{ formatCount(points.sampling.original_rows) }} rows (seed
          {{ points.sampling.seed }})</template
        >.
      </p>
      <DsbfChart
        :theme="theme"
        :option="option"
        :empty="!points.values.length"
        empty-message="No row has a value in both columns."
        :export-name="`${first}-${second}`"
      />
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../../api/client'
import { formatCount } from '../../format'
import type { components } from '../../generated/api'
import { useThemeStore } from '../../stores/theme'
import DsbfChart from '../charts/DsbfChart.vue'
import { pairOption, plotKind, type PairPoints } from './relationships'

const props = defineProps<{
  runId: string
  columns: components['schemas']['ColumnSummary'][]
  first: string
  second: string
}>()
defineEmits<{ select: [first: string, second: string] }>()

const theme = computed(() => useThemeStore().theme)
const points = ref<PairPoints | null>(null)
const error = ref('')

const typeOf = (name: string) => props.columns.find((c) => c.name === name)?.semantic_type ?? ''
const kind = computed(() => plotKind(typeOf(props.first), typeOf(props.second)))
const option = computed(() =>
  points.value ? pairOption(kind.value, points.value, typeOf(props.first) === 'numeric') : {},
)

watch(
  () => [props.runId, props.first, props.second],
  async () => {
    points.value = null
    error.value = ''
    if (!props.first || !props.second || props.first === props.second) return
    try {
      points.value = (await unwrap(
        api.GET('/v2/runs/{run_id}/snapshot/pair', {
          params: {
            path: { run_id: props.runId },
            query: { first: props.first, second: props.second },
          },
        }),
      )) as PairPoints
    } catch (caught) {
      error.value = caught instanceof Error ? caught.message : String(caught)
    }
  },
  { immediate: true },
)
</script>
