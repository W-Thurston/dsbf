<template>
  <div class="stack">
    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="matrix === null || columns === null" class="loading">Loading…</p>
    <template v-else>
      <SummaryCard
        :matrix="matrix"
        :chips="chips"
        :active="chip"
        :outliers="outliers"
        @toggle="toggleChip"
      />

      <template v-if="matrix.cells.length">
        <section v-for="panel in panels" :key="panel.title" class="stack">
          <p v-if="panel.columns.length < panel.total" class="muted" role="status">
            {{ panel.title }}: {{ panel.columns.length }} of {{ panel.total }} columns, strongest
            first; the table below lists every pair.
          </p>
          <DsbfChart
            :title="panel.title"
            :theme="theme"
            :option="matrixOption(panel)"
            :height="`${panelHeight(panel)}px`"
            :export-name="`matrix-${panel.title}`"
          />
        </section>

        <AssociationsPanel
          :matrix="matrix"
          :columns="columns"
          :column="column"
          :restrict="restrict"
          @select-column="chooseColumn"
        />
      </template>

      <PairDetail
        :run-id="runId"
        :columns="columns"
        :matrix="matrix"
        :first="first"
        :second="second"
        :outputs="outputs"
        @select="choose"
      />
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, unwrap } from '../api/client'
import DsbfChart from '../components/charts/DsbfChart.vue'
import AssociationsPanel from '../components/relationships/AssociationsPanel.vue'
import PairDetail from '../components/relationships/PairDetail.vue'
import SummaryCard, { type Outliers } from '../components/relationships/SummaryCard.vue'
import {
  capPanel,
  chipsOf,
  matrixOption,
  panelHeight,
  panelsOf,
  type Chip,
} from '../components/relationships/relationships'
import type { components } from '../generated/api'
import { useThemeStore } from '../stores/theme'

const props = defineProps<{ runId: string }>()

type Output = Record<string, unknown>
const CHECKS = [
  'collinearity',
  'duplicate_columns',
  'target_leakage',
  'multivariate_outliers',
  'contingency',
  'group_differences',
]

const route = useRoute()
const router = useRouter()
const theme = computed(() => useThemeStore().theme)
const matrix = ref<components['schemas']['RelationshipMatrix'] | null>(null)
const columns = ref<components['schemas']['ColumnSummary'][] | null>(null)
const outputs = ref<Record<string, Output | null>>({})
const error = ref('')

// The browser's column, the chosen pair and the active chip live in the URL, so a view can
// be shared and survives a reload.
const first = computed(() => String(route.query.first ?? ''))
const second = computed(() => String(route.query.second ?? ''))
const column = computed(() => String(route.query.column ?? ''))
const chip = computed(() => String(route.query.chip ?? ''))
function replaceQuery(patch: Record<string, string | undefined>) {
  void router.replace({ query: { ...route.query, ...patch } })
}
const choose = (a: string, b: string) =>
  replaceQuery({ first: a || undefined, second: b || undefined })
const chooseColumn = (name: string) =>
  replaceQuery({ column: name === column.value ? undefined : name })
const toggleChip = (id: Chip['id']) => replaceQuery({ chip: id === chip.value ? undefined : id })

const panels = computed(() => (matrix.value ? panelsOf(matrix.value).map((p) => capPanel(p)) : []))
const chips = computed(() =>
  chipsOf({
    collinearity: outputs.value.collinearity,
    duplicate_columns: outputs.value.duplicate_columns,
    target_leakage: outputs.value.target_leakage,
  }),
)
const restrict = computed(() => {
  const active = chips.value.find((c) => c.id === chip.value)
  return active ? new Set(active.columns) : null
})
const outliers = computed<Outliers | null>(() => {
  const o = outputs.value.multivariate_outliers
  return o && Array.isArray(o.feature_columns) ? (o as unknown as Outliers) : null
})

const failure = (caught: unknown) =>
  (error.value = caught instanceof Error ? caught.message : String(caught))
const path = { params: { path: { run_id: props.runId } } }
unwrap(api.GET('/v2/runs/{run_id}/relationships', path))
  .then((m) => (matrix.value = m))
  .catch(failure)
unwrap(api.GET('/v2/runs/{run_id}/columns', path))
  .then((list) => (columns.value = list))
  .catch(failure)
// A check that did not run, or ran without output, leaves its card out rather than failing the page.
for (const id of CHECKS)
  unwrap(
    api.GET('/v2/runs/{run_id}/checks/{check_id}', {
      params: { path: { run_id: props.runId, check_id: id } },
    }),
  )
    .then(
      (r) =>
        (outputs.value = { ...outputs.value, [id]: r.status === 'ok' ? (r.output ?? null) : null }),
    )
    .catch(() => undefined)
</script>
