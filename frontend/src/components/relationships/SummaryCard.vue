<template>
  <section class="card" aria-labelledby="relationships-title" data-section="relationships-summary">
    <h1 id="relationships-title" class="run-name">Relationships</h1>
    <p class="muted">
      Each cell is the statistic one check measured for that pair; an empty square was not tested.
    </p>
    <template v-if="matrix.cells.length">
      <div
        class="strength"
        role="img"
        :aria-label="`${strengths.map((s) => `${counts[s]} ${s}`).join(', ')} pairs`"
        data-role="strength-bar"
      >
        <span
          v-for="s in strengths"
          :key="s"
          :class="['band', s]"
          :style="{ flexGrow: counts[s] }"
          :title="`${counts[s]} ${s}`"
        />
      </div>
      <ul class="legend">
        <li v-for="s in strengths" :key="s">
          <span :class="['swatch', s]" />{{ formatCount(counts[s]) }} {{ s }}
        </li>
      </ul>
      <p v-if="unbanded" class="muted">
        {{ formatCount(unbanded) }} tested {{ unbanded === 1 ? 'pair names' : 'pairs name' }} no
        strength band and {{ unbanded === 1 ? 'is' : 'are' }} left out of the bar.
      </p>
    </template>
    <p v-else class="muted" role="status">No pair was tested in this run.</p>

    <div v-if="chips.length" class="chips" role="group" aria-label="Warnings">
      <button
        v-for="chip in chips"
        :key="chip.id"
        type="button"
        class="chip"
        :aria-pressed="active === chip.id"
        :data-chip="chip.id"
        @click="$emit('toggle', chip.id)"
      >
        {{ chip.label }} · {{ chip.columns.length }}
      </button>
    </div>

    <p v-if="outliers" class="muted" data-role="multivariate-outliers">
      Multivariate outliers: {{ formatCount(outliers.far_rows) }} of
      {{ formatCount(outliers.rows_used) }} rows stand out across
      {{ plural(outliers.feature_columns.length, 'numeric column') }} together<template
        v-if="!outliers.far_rows"
      >
        (none beyond the reference range)</template
      >.
    </p>

    <ul v-if="matrix.coverage.length" class="coverage" aria-label="Coverage">
      <li v-for="c in matrix.coverage" :key="c.check_id">
        <code>{{ c.check_id }}</code
        >: {{ c.status === 'ok' ? plural(c.pairs_tested, 'pair') + ' tested' : c.status
        }}<template v-if="Object.keys(c.untested).length">
          ({{ untested(c.untested) }} not tested)</template
        >
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { formatCount, plural } from '../../format'
import type { components } from '../../generated/api'
import { STRENGTHS, strengthCounts, type Chip } from './relationships'

export interface Outliers {
  far_rows: number
  rows_used: number
  feature_columns: string[]
}

const props = defineProps<{
  matrix: components['schemas']['RelationshipMatrix']
  chips: Chip[]
  active: string
  outliers: Outliers | null
}>()
defineEmits<{ toggle: [id: Chip['id']] }>()

const strengths = STRENGTHS
const tallied = computed(() => strengthCounts(props.matrix.cells))
const counts = computed(() => tallied.value.counts)
const unbanded = computed(() => tallied.value.unbanded)

const untested = (reasons: Record<string, number>) =>
  Object.entries(reasons)
    .map(([reason, n]) => `${n} ${reason.replaceAll('_', ' ')}`)
    .join(', ')
</script>

<style scoped>
.run-name {
  font-size: 22px;
}
.strength {
  display: flex;
  height: 14px;
  border-radius: 7px;
  overflow: hidden;
  background: var(--border);
  gap: 1px;
}
.band.strong,
.swatch.strong {
  background: var(--level-red, #f87171);
}
.band.moderate,
.swatch.moderate {
  background: var(--level-amber, #fbbf24);
}
.band.weak,
.swatch.weak {
  background: #60a5fa;
}
.band.negligible,
.swatch.negligible {
  background: var(--border);
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  list-style: none;
  padding: 0;
  margin: 8px 0;
}
.swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  margin-right: 6px;
  border: 1px solid var(--border);
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 8px 0;
}
.chip {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 2px 12px;
  background: var(--bg);
  color: inherit;
  cursor: pointer;
}
.chip[aria-pressed='true'] {
  background: var(--level-amber, #fbbf24);
  color: #111;
}
.coverage {
  margin: 8px 0 0;
  padding-left: 18px;
}
</style>
