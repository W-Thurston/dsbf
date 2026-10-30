<template>
  <section v-if="selected" class="stack" data-section="pair-detail">
    <div class="card">
      <h2 class="card-title">{{ first }} × {{ second }}</h2>
      <p v-if="!cells.length" class="muted">No check tested this pair.</p>
      <dl v-else class="strip" aria-label="Statistics">
        <div v-for="cell in cells" :key="cell.statistic">
          <dt>{{ STATISTIC_LABEL[cell.statistic] }}</dt>
          <dd>
            {{ cell.value.toFixed(3)
            }}<template v-if="cell.interval">
              ({{ cell.interval.low.toFixed(2) }} to {{ cell.interval.high.toFixed(2) }})</template
            >
            · {{ cell.strength ?? (cell.flagged ? 'flagged' : 'not banded') }} · n
            {{ formatCount(cell.n_used) }}
          </dd>
        </div>
      </dl>
      <ul v-if="warnings.length" class="warnings" aria-label="Warnings">
        <li v-for="w in warnings" :key="w">{{ w }}</li>
      </ul>
      <template v-if="counts">
        <table class="data-table" aria-label="Counts">
          <thead>
            <tr>
              <th scope="col">{{ counts.first }} ↓ / {{ counts.second }} →</th>
              <th v-for="level in levels" :key="level" scope="col">{{ level }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, name) in counts.table" :key="name">
              <th scope="row">{{ name }}</th>
              <td v-for="level in levels" :key="level">{{ row[level] ?? 0 }}</td>
            </tr>
          </tbody>
        </table>
        <p v-if="counts.truncated" class="muted">Only the most frequent levels are shown.</p>
      </template>
      <p v-if="difference" class="muted">
        {{ difference.numeric }} by {{ difference.categorical }}: ω²
        {{ difference.omega_squared.toFixed(3) }}, ε² {{ difference.epsilon_squared.toFixed(3) }};
        highest mean rank in <strong>{{ difference.top_group }}</strong
        >, lowest in <strong>{{ difference.bottom_group }}</strong
        >.
      </p>
    </div>
    <PairPlot
      :run-id="runId"
      :columns="columns"
      :first="first"
      :second="second"
      @select="(a, b) => $emit('select', a, b)"
    />
  </section>
  <section v-else class="card" data-section="pair-detail">
    <h2 class="card-title">Pair detail</h2>
    <p class="muted">Choose a pair from the table to see its statistics and plot.</p>
    <PairPlot
      :run-id="runId"
      :columns="columns"
      :first="first"
      :second="second"
      @select="(a, b) => $emit('select', a, b)"
    />
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { formatCount } from '../../format'
import type { components } from '../../generated/api'
import PairPlot from './PairPlot.vue'
import { pairIn, rowsOf, STATISTIC_LABEL, type LeakageRow, type VifRow } from './relationships'

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

const props = defineProps<{
  runId: string
  columns: components['schemas']['ColumnSummary'][]
  matrix: components['schemas']['RelationshipMatrix']
  first: string
  second: string
  outputs: Record<string, Record2 | null>
}>()
defineEmits<{ select: [first: string, second: string] }>()

const selected = computed(() => !!props.first && !!props.second && props.first !== props.second)
const cells = computed(() =>
  props.matrix.cells.filter(
    (c) =>
      (c.first === props.first && c.second === props.second) ||
      (c.first === props.second && c.second === props.first),
  ),
)
const pair = (key: string, keys: [string, string]) =>
  pairIn(rowsOf<Record2>(props.outputs[key] ?? {}, 'pairs'), keys, props.first, props.second)
const counts = computed(() => pair('contingency', ['first', 'second']) as Contingency | undefined)
const difference = computed(
  () => pair('group_differences', ['numeric', 'categorical']) as Difference | undefined,
)
const levels = computed(() => [
  ...new Set(Object.values(counts.value?.table ?? {}).flatMap((row) => Object.keys(row))),
])

// What the other checks say about either column, so a strong pair is read with its caveats.
const warnings = computed(() => {
  const out: string[] = []
  const vifs = rowsOf<VifRow>(props.outputs.collinearity ?? {}, 'columns')
  for (const name of [props.first, props.second]) {
    const row = vifs.find((r) => r.column === name)
    if (row && row.vif >= 10) out.push(`${name} has a VIF of ${row.vif.toFixed(1)} (${row.tier}).`)
  }
  for (const group of rowsOf<string[]>(props.outputs.duplicate_columns ?? {}, 'groups'))
    if (group.includes(props.first) && group.includes(props.second))
      out.push(`${props.first} and ${props.second} hold equal values.`)
  for (const row of rowsOf<LeakageRow>(props.outputs.target_leakage ?? {}, 'features'))
    if (row.flagged && (row.feature === props.first || row.feature === props.second))
      out.push(
        `${row.feature} was flagged for possible target leakage (${row.metric} ${row.value.toFixed(3)}).`,
      )
  return out
})
</script>

<style scoped>
.strip {
  display: grid;
  gap: 4px;
  margin: 0 0 8px;
}
.strip div {
  display: flex;
  gap: 12px;
}
dt {
  min-width: 190px;
  font-weight: 600;
}
dd {
  margin: 0;
}
.warnings {
  margin: 8px 0;
  padding-left: 18px;
}
</style>
