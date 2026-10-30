<template>
  <CheckCard
    v-slot="{ output }"
    :run-id="runId"
    check-id="fuzzy_duplicates"
    title="Near-duplicate rows"
  >
    <template v-for="f in [asFuzzy(output)]" :key="f.pairs_flagged">
      <p v-if="!f.top_pairs.length" class="muted" data-role="no-pairs">
        No pair of rows is similar enough to flag (threshold {{ f.threshold }}).
      </p>
      <ul v-else class="pairs">
        <li v-for="(pair, index) in f.top_pairs" :key="index" data-role="pair">
          <span class="track"
            ><span class="fill" :style="{ width: `${pair.similarity * 100}%` }"
          /></span>
          <span class="value">{{ percent(pair.similarity) }}</span>
          <span class="rows muted">rows {{ pair.first.row }} and {{ pair.second.row }}</span>
          <span class="diff">
            <template v-for="column in pair.differing_columns" :key="column">
              <code>{{ column }}</code
              >:
              {{ (pair.values[column] ?? []).join(' ≠ ') }}
            </template>
          </span>
        </li>
      </ul>
      <p class="muted">
        {{ f.pairs_flagged }} pairs flagged at similarity {{ f.threshold }} across
        {{ f.rows_involved }} rows; {{ percent(f.coverage) }} of the possible pairs were compared.
      </p>
    </template>
  </CheckCard>
</template>

<script setup lang="ts">
import CheckCard from '../relationships/CheckCard.vue'
import { percent } from './quality'

defineProps<{ runId: string }>()

interface Pair {
  first: { row: number }
  second: { row: number }
  similarity: number
  differing_columns: string[]
  values: Record<string, (string | number | null)[]>
}
interface Fuzzy {
  top_pairs: Pair[]
  pairs_flagged: number
  rows_involved: number
  threshold: number
  coverage: number
}

const asFuzzy = (output: unknown) => output as Fuzzy
</script>

<style scoped>
.pairs {
  list-style: none;
  padding: 0;
  margin: 0;
}
.pairs li {
  display: grid;
  grid-template-columns: 120px 44px 120px 1fr;
  gap: 10px;
  align-items: center;
  padding: 3px 0;
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
  background: var(--chart-primary);
}
.value {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.diff {
  overflow-wrap: anywhere;
}
</style>
