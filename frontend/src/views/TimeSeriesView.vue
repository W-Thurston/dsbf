<template>
  <div class="stack">
    <section class="card" aria-labelledby="ts-title">
      <h1 id="ts-title" class="run-name">Time series</h1>
      <p v-if="state === 'loading'" class="loading">Loading…</p>
    </section>

    <section
      v-if="state === 'empty'"
      class="card"
      aria-labelledby="empty-title"
      data-role="empty-state"
    >
      <h2 id="empty-title" class="card-title">No time series was analysed</h2>
      <p>{{ emptyReason }}</p>
      <p v-if="noIndex">
        Set the datetime column that orders the rows: run
        <code>dsbf profile &lt;data&gt; --time-index &lt;column&gt;</code>, or set
        <code>time_index</code> in the profile config. A rerun from the dashboard comes with the
        settings panel.
      </p>
    </section>

    <template v-else-if="state === 'ready'">
      <section class="card" aria-label="Series" data-role="series">
        <p class="muted">
          Ordered by <code>{{ series.index }}</code
          >, as {{ plural(series.periods_total, `${series.frequency} period`)
          }}<template v-if="series.truncated"
            >; the latest {{ series.max_periods }} are used</template
          >, each the {{ series.aggregate }} of its rows.
        </p>
        <label>
          Column
          <select v-model="column" data-role="column">
            <option v-for="name in names" :key="name" :value="name">{{ name }}</option>
          </select>
        </label>
        <ul v-if="notRun.length" class="muted" data-role="not-run">
          <li v-for="item in notRun" :key="item.id">{{ item.id }}: {{ item.reason }}</li>
        </ul>
      </section>

      <StationarityTable
        v-if="stationarity"
        :output="stationarity"
        :column="column"
        @select="column = $event"
      />
      <AutocorrelationPlots v-if="autocorrelation" :output="autocorrelation" :column="column" />
      <DecompositionPlots
        v-if="decomposition"
        :run-id="runId"
        :output="decomposition"
        :column="column"
      />
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../api/client'
import AutocorrelationPlots from '../components/timeseries/AutocorrelationPlots.vue'
import DecompositionPlots from '../components/timeseries/DecompositionPlots.vue'
import StationarityTable from '../components/timeseries/StationarityTable.vue'
import {
  TIME_SERIES_CHECKS,
  type AutocorrelationOutput,
  type DecompositionOutput,
  type SeriesOutput,
  type StationarityOutput,
} from '../components/timeseries/timeseries'
import { plural } from '../format'
import type { components } from '../generated/api'

type Result = components['schemas']['CheckResult']

const props = defineProps<{ runId: string }>()

const results = ref<Record<string, Result>>({})
const state = ref<'loading' | 'empty' | 'ready'>('loading')
const column = ref('')

const outputOf = <T,>(id: string) =>
  results.value[id]?.status === 'ok' ? (results.value[id].output as T | null) : null
const stationarity = computed(() => outputOf<StationarityOutput>('stationarity'))
const autocorrelation = computed(() => outputOf<AutocorrelationOutput>('autocorrelation'))
const decomposition = computed(() => outputOf<DecompositionOutput>('seasonal_decomposition'))

const series = computed<SeriesOutput>(
  () => (stationarity.value ?? autocorrelation.value ?? decomposition.value) as SeriesOutput,
)
const names = computed(() => {
  const found = new Set<string>()
  for (const output of [stationarity.value, autocorrelation.value, decomposition.value])
    for (const name of Object.keys(output?.columns ?? {})) found.add(name)
  return [...found]
})

const firstNote = computed(
  () => Object.values(results.value).find((r) => r.notes.length)?.notes[0] ?? null,
)
const noIndex = computed(() => firstNote.value?.code === 'no_time_index')
const emptyReason = computed(
  () => firstNote.value?.message ?? 'None of the three time series checks ran in this run.',
)
const notRun = computed(() =>
  TIME_SERIES_CHECKS.filter((id) => !outputOf(id)).map((id) => ({
    id,
    reason: results.value[id]?.notes[0]?.message ?? 'it did not run in this run',
  })),
)

watch(
  () => props.runId,
  async () => {
    state.value = 'loading'
    const loaded: Record<string, Result> = {}
    await Promise.all(
      TIME_SERIES_CHECKS.map(async (id) => {
        try {
          loaded[id] = await unwrap(
            api.GET('/v2/runs/{run_id}/checks/{check_id}', {
              params: { path: { run_id: props.runId, check_id: id } },
            }),
          )
        } catch {
          // A check that did not run is the same as one that stopped: it has no result.
        }
      }),
    )
    results.value = loaded
    column.value = names.value[0] ?? ''
    state.value = names.value.length ? 'ready' : 'empty'
  },
  { immediate: true },
)
</script>
