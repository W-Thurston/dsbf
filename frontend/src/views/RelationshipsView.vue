<template>
  <div class="stack">
    <section class="card" aria-labelledby="relationships-title">
      <p class="muted crumbs">
        <router-link to="/">Datasets</router-link> /
        <router-link :to="{ name: 'run', params: { runId } }">Run</router-link> /
      </p>
      <h1 id="relationships-title" class="run-name">Relationships</h1>
      <p class="muted">
        Each cell is the statistic one check measured for that pair; an empty square was not tested.
      </p>
      <ul v-if="matrix" class="coverage" aria-label="Coverage">
        <li v-for="c in matrix.coverage" :key="c.check_id">
          <code>{{ c.check_id }}</code
          >: {{ c.status === 'ok' ? plural(c.pairs_tested, 'pair') + ' tested' : c.status
          }}<template v-if="Object.keys(c.untested).length">
            ({{ untested(c.untested) }} not tested)</template
          >
        </li>
      </ul>
    </section>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="matrix === null || columns === null" class="loading">Loading…</p>
    <template v-else>
      <p v-if="!matrix.cells.length" class="muted card" role="status">
        No pair was tested in this run.
      </p>
      <template v-else>
        <DsbfChart
          v-for="panel in panels"
          :key="panel.title"
          :title="panel.title"
          :theme="theme"
          :option="matrixOption(panel)"
          :height="`${Math.max(panel.columns.length * 28 + 140, 260)}px`"
          :export-name="`matrix-${panel.title}`"
        />

        <section class="card" aria-labelledby="pairs-title">
          <h2 id="pairs-title" class="card-title">Tested pairs</h2>
          <p class="muted">Grouped by statistic, then strongest first.</p>
          <table class="data-table">
            <thead>
              <tr>
                <th scope="col">Statistic</th>
                <th scope="col">Pair</th>
                <th scope="col">Value</th>
                <th scope="col">Strength</th>
                <th scope="col">n</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="cell in cells" :key="`${cell.check_id}|${cell.first}|${cell.second}`">
                <td>{{ STATISTIC_LABEL[cell.statistic] }}</td>
                <td>
                  <router-link :to="{ query: { first: cell.first, second: cell.second } }"
                    >{{ cell.first }} × {{ cell.second }}</router-link
                  >
                </td>
                <td>
                  {{ cell.value.toFixed(3)
                  }}<template v-if="cell.interval">
                    ({{ cell.interval.low.toFixed(2) }} to
                    {{ cell.interval.high.toFixed(2) }})</template
                  >
                </td>
                <td>{{ cell.strength ?? (cell.flagged ? 'flagged' : '') }}</td>
                <td>{{ formatCount(cell.n_used) }}</td>
              </tr>
            </tbody>
          </table>
        </section>
      </template>

      <PairPlot
        :run-id="runId"
        :columns="columns"
        :first="first"
        :second="second"
        @select="choose"
      />

      <CheckCard v-slot="{ output }" :run-id="runId" check-id="collinearity" title="Collinearity">
        <p class="muted">
          {{ vifs(output).filter((r) => r.vif >= 10).length }} of {{ vifs(output).length }} columns
          have a VIF of 10 or more.
        </p>
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">Column</th>
              <th scope="col">VIF</th>
              <th scope="col">Tier</th>
              <th scope="col">Most correlated with</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in vifs(output)" :key="row.column">
              <td>{{ row.column }}</td>
              <td>{{ row.vif.toFixed(1) }}</td>
              <td>{{ row.tier }}</td>
              <td>{{ row.partners.map((p) => `${p.column} (r ${p.r.toFixed(2)})`).join(', ') }}</td>
            </tr>
          </tbody>
        </table>
      </CheckCard>

      <CheckCard
        v-slot="{ output }"
        :run-id="runId"
        check-id="target_leakage"
        title="Target leakage"
      >
        <p class="muted">
          Features measured against <code>{{ output.target }}</code
          >; a high value is not proof of leakage.
        </p>
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">Feature</th>
              <th scope="col">Metric</th>
              <th scope="col">Value</th>
              <th scope="col">Flagged</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in leaks(output)" :key="`${row.feature}|${row.metric}`">
              <td>{{ row.feature }}</td>
              <td>{{ row.metric }}</td>
              <td>{{ row.value.toFixed(3) }}</td>
              <td>{{ row.flagged ? 'yes' : '' }}</td>
            </tr>
          </tbody>
        </table>
      </CheckCard>

      <CheckCard
        v-slot="{ output }"
        :run-id="runId"
        check-id="contingency"
        title="Contingency table"
      >
        <template v-if="selected && contingency(output)">
          <table class="data-table" aria-label="Counts">
            <thead>
              <tr>
                <th scope="col">
                  {{ contingency(output)!.first }} ↓ / {{ contingency(output)!.second }} →
                </th>
                <th v-for="level in levels(contingency(output)!.table)" :key="level" scope="col">
                  {{ level }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(counts, row) in contingency(output)!.table" :key="row">
                <th scope="row">{{ row }}</th>
                <td v-for="level in levels(contingency(output)!.table)" :key="level">
                  {{ counts[level] ?? 0 }}
                </td>
              </tr>
            </tbody>
          </table>
          <p v-if="contingency(output)!.truncated" class="muted">
            Only the most frequent levels are shown.
          </p>
        </template>
        <p v-else class="muted">Select a pair of categorical columns above to see its counts.</p>
      </CheckCard>

      <CheckCard
        v-slot="{ output }"
        :run-id="runId"
        check-id="group_differences"
        title="Group differences"
      >
        <p v-if="selected && difference(output)" class="muted">
          {{ difference(output)!.numeric }} by {{ difference(output)!.categorical }}: ω²
          {{ difference(output)!.omega_squared.toFixed(3) }}, ε²
          {{ difference(output)!.epsilon_squared.toFixed(3) }}; highest mean rank in
          <strong>{{ difference(output)!.top_group }}</strong
          >, lowest in <strong>{{ difference(output)!.bottom_group }}</strong
          >.
        </p>
        <p v-else class="muted">
          Select a numeric and a categorical column above to see how the groups differ.
        </p>
      </CheckCard>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, unwrap } from '../api/client'
import DsbfChart from '../components/charts/DsbfChart.vue'
import CheckCard from '../components/relationships/CheckCard.vue'
import PairPlot from '../components/relationships/PairPlot.vue'
import {
  matrixOption,
  pairIn,
  panelsOf,
  rowsOf,
  sortedCells,
  STATISTIC_LABEL,
  type LeakageRow,
  type VifRow,
} from '../components/relationships/relationships'
import { formatCount, plural } from '../format'
import type { components } from '../generated/api'
import { useThemeStore } from '../stores/theme'

const props = defineProps<{ runId: string }>()

type Output = Record<string, unknown>
type Record2 = Record<string, unknown>
interface Contingency extends Record2 {
  first: string
  second: string
  table: Record<string, Record<string, number>>
  truncated: boolean
}
interface Difference extends Record2 {
  numeric: string
  categorical: string
  omega_squared: number
  epsilon_squared: number
  top_group: string
  bottom_group: string
}

const route = useRoute()
const router = useRouter()
const theme = computed(() => useThemeStore().theme)
const matrix = ref<components['schemas']['RelationshipMatrix'] | null>(null)
const columns = ref<components['schemas']['ColumnSummary'][] | null>(null)
const error = ref('')

// The chosen pair lives in the URL, so a plot can be shared and survives a reload.
const first = computed(() => String(route.query.first ?? ''))
const second = computed(() => String(route.query.second ?? ''))
const selected = computed(() => !!first.value && !!second.value && first.value !== second.value)
function choose(a: string, b: string) {
  void router.replace({ query: { first: a || undefined, second: b || undefined } })
}

const panels = computed(() => (matrix.value ? panelsOf(matrix.value) : []))
const cells = computed(() => (matrix.value ? sortedCells(matrix.value) : []))

const untested = (reasons: Record<string, number>) =>
  Object.entries(reasons)
    .map(([reason, n]) => `${n} ${reason.replaceAll('_', ' ')}`)
    .join(', ')
const vifs = (output: Output) => rowsOf<VifRow>(output, 'columns')
const leaks = (output: Output) => rowsOf<LeakageRow>(output, 'features')
const contingency = (output: Output) =>
  pairIn(rowsOf<Contingency>(output, 'pairs'), ['first', 'second'], first.value, second.value)
const difference = (output: Output) =>
  pairIn(rowsOf<Difference>(output, 'pairs'), ['numeric', 'categorical'], first.value, second.value)
const levels = (table: Record<string, Record<string, number>>) => [
  ...new Set(Object.values(table).flatMap((counts) => Object.keys(counts))),
]

const failure = (caught: unknown) =>
  (error.value = caught instanceof Error ? caught.message : String(caught))
const path = { params: { path: { run_id: props.runId } } }
unwrap(api.GET('/v2/runs/{run_id}/relationships', path))
  .then((m) => (matrix.value = m))
  .catch(failure)
unwrap(api.GET('/v2/runs/{run_id}/columns', path))
  .then((list) => (columns.value = list))
  .catch(failure)
</script>

<style scoped>
.run-name {
  font-size: 22px;
}
.crumbs {
  margin: 0 0 4px;
}
.coverage {
  margin: 0;
  padding-left: 18px;
}
</style>
