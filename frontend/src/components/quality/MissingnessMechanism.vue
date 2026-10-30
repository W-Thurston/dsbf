<template>
  <CheckCard
    v-slot="{ output }"
    :run-id="runId"
    check-id="missingness_mechanism"
    title="Why values are missing"
  >
    <template v-for="m in [asMechanism(output)]" :key="m.tests_run">
      <p v-if="!rows(m).length" class="muted" data-role="no-mechanism">
        No column has enough missing values to test.
      </p>
      <div v-for="[column, c] in rows(m)" :key="column" class="column" :data-column="column">
        <h3 class="name">
          <router-link :to="{ name: 'column', params: { runId, column } }">{{
            column
          }}</router-link>
          <span class="muted"> missing in {{ percent(c.null_share) }} of rows</span>
          <span
            :class="['badge', c.assessment === 'depends_on_observed' ? 'badge-warn' : 'badge-info']"
          >
            {{ ASSESSMENT[c.assessment] }}
          </span>
        </h3>
        <p v-if="c.reason" class="muted">{{ c.reason }}</p>
        <ul
          v-if="c.dependences.length"
          class="bars"
          :aria-label="`What ${column}'s missingness depends on`"
        >
          <li v-for="p in c.dependences" :key="p.column" :data-predictor="p.column">
            <span class="label">{{ p.column }}</span>
            <span class="track"
              ><span class="fill" :style="{ width: `${effectBar(p).width * 100}%` }"
            /></span>
            <span class="value">{{ effectBar(p).text }}</span>
            <span class="muted kind">{{ EFFECT[p.effect_name] }}</span>
          </li>
        </ul>
        <p v-if="Object.keys(c.missing_with).length" class="muted">
          Goes missing together with {{ Object.keys(c.missing_with).join(', ') }}.
        </p>
      </div>
      <p class="muted">
        {{ m.tests_flagged }} of {{ m.tests_run }} tests flagged a dependence after
        {{ m.correction }} correction.
      </p>
    </template>
  </CheckCard>
</template>

<script setup lang="ts">
import CheckCard from '../relationships/CheckCard.vue'
import { effectBar, percent, type Predictor } from './quality'

defineProps<{ runId: string }>()

interface ColumnMechanism {
  null_share: number
  assessment: 'depends_on_observed' | 'no_dependence_found' | 'untestable'
  reason: string | null
  dependences: Predictor[]
  missing_with: Record<string, unknown>
}
interface Mechanism {
  columns: Record<string, ColumnMechanism>
  tests_run: number
  tests_flagged: number
  correction: string
}

const ASSESSMENT = {
  depends_on_observed: 'Depends on other columns',
  no_dependence_found: 'No dependence found',
  untestable: 'Could not be tested',
}
const EFFECT = { rank_biserial: 'rank-biserial r', cramers_v: 'Cramér’s V' }

const asMechanism = (output: unknown) => output as Mechanism
// The columns that depend on others first, then by how much is missing.
const rows = (m: Mechanism) =>
  Object.entries(m.columns).sort(
    ([, a], [, b]) =>
      Number(b.dependences.length > 0) - Number(a.dependences.length > 0) ||
      b.null_share - a.null_share,
  )
</script>

<style scoped>
.column {
  margin: 12px 0;
}
.name {
  font-size: 14px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 10px;
}
.bars {
  list-style: none;
  padding: 0;
  margin: 6px 0;
}
.bars li {
  display: grid;
  grid-template-columns: 140px 1fr 48px 110px;
  gap: 10px;
  align-items: center;
  padding: 2px 0;
}
.label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.track {
  height: 10px;
  border-radius: 5px;
  background: var(--border);
  overflow: hidden;
}
.fill {
  display: block;
  height: 100%;
  background: var(--level-amber);
}
.value {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.kind {
  font-size: 12px;
}
</style>
