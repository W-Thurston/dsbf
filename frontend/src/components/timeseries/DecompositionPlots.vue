<template>
  <section class="card" aria-labelledby="stl-title" data-section="decomposition">
    <h2 id="stl-title" class="card-title">Trend and seasonality (STL)</h2>
    <p v-if="!info" class="muted">{{ column }} was not decomposed: {{ skipped }}</p>
    <template v-else>
      <p class="muted" data-role="strengths">
        Cycle of {{ info.period }} periods ({{ output.period_source }}). Trend strength
        {{ info.trend_strength.toFixed(2) }} ({{
          strengthWords(info.trend_strength, info.trend_floor)
        }}), seasonal strength {{ info.seasonal_strength.toFixed(2) }} ({{
          strengthWords(info.seasonal_strength, info.seasonal_floor)
        }}); the remainder is {{ Math.round(info.remainder_share * 100) }}% of the variance. White
        noise of this length reaches {{ info.trend_floor.toFixed(2) }} (trend) and
        {{ info.seasonal_floor.toFixed(2) }} (seasonal).
      </p>
      <p class="muted">
        Period 0 starts {{ info.series_start }}; each period is one {{ output.frequency }} step.
      </p>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
      <p v-else-if="!components" class="loading">Loading…</p>
      <DsbfChart
        v-else
        title="Observed, trend, seasonal and remainder"
        :theme="theme"
        :option="stlOption(components)"
        height="560px"
        :export-name="`${column}-stl`"
      />
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../../api/client'
import { useThemeStore } from '../../stores/theme'
import DsbfChart from '../charts/DsbfChart.vue'
import {
  ARTIFACT_PAGE,
  componentsOf,
  rowRange,
  stlOption,
  strengthWords,
  type Components,
  type DecompositionOutput,
} from './timeseries'

const props = defineProps<{ runId: string; output: DecompositionOutput; column: string }>()

const REF = 'seasonal_decomposition/decomposition'
const theme = computed(() => useThemeStore().theme)
const info = computed(() => props.output.columns[props.column])
const skipped = computed(() => props.output.skipped[props.column] ?? 'it is not a series column.')
const components = ref<Components | null>(null)
const error = ref('')

watch(
  () => [props.runId, props.column, props.output],
  async () => {
    components.value = null
    error.value = ''
    const range = rowRange(props.output.columns, props.column)
    if (!range) return
    try {
      const names: string[] = []
      const rows: unknown[][] = []
      // The artifact is paged; a column's series is a contiguous range of its rows.
      for (let at = 0; at < range.count; at += ARTIFACT_PAGE) {
        const page = await unwrap(
          api.GET('/v2/runs/{run_id}/artifacts/{ref}', {
            params: {
              path: { run_id: props.runId, ref: REF },
              query: {
                offset: range.offset + at,
                limit: Math.min(ARTIFACT_PAGE, range.count - at),
              },
            },
          }),
        )
        if (!('rows' in page)) throw new Error('the decomposition is not a table')
        names.splice(0, names.length, ...page.columns)
        rows.push(...page.rows.map((row) => [...row]))
      }
      components.value = componentsOf(names, rows, props.column)
    } catch (caught) {
      error.value = caught instanceof Error ? caught.message : String(caught)
    }
  },
  { immediate: true },
)
</script>
