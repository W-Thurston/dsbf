<template>
  <section class="card" aria-labelledby="missing-title" data-section="missingness">
    <h2 id="missing-title" class="card-title">Missing values</h2>
    <p v-if="!withNulls.length" class="muted" data-role="no-nulls">
      No column has a missing value.
    </p>
    <template v-else>
      <BarList :items="withNulls" label="Missing values per column" />
      <template v-if="heat">
        <h3 class="sub">Columns that go missing together</h3>
        <NullMatrix
          :labels="heat.labels"
          :values="heat.values"
          label="Share of rows null in either column that are null in both"
        />
        <p class="muted note">
          A cell is the share of rows null in either column that are null in both.<template
            v-if="
              facts.missingness &&
              facts.missingness.columns_considered < facts.missingness.columns_with_nulls
            "
          >
            Only the {{ facts.missingness.columns_considered }} columns with the most nulls are
            paired.</template
          >
        </p>
      </template>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { components } from '../../generated/api'
import BarList from './BarList.vue'
import NullMatrix from './NullMatrix.vue'
import { MAX_HEAT_COLUMNS } from './overview'

const props = defineProps<{
  columns: components['schemas']['ColumnSummary'][]
  facts: components['schemas']['DatasetFacts']
}>()

/** Columns with nulls, the most missing first, as a count of rows. */
const withNulls = computed(() =>
  props.columns
    .filter((c) => c.counts.nulls > 0)
    .map((c) => ({ label: c.name, count: c.counts.nulls }))
    .sort((a, b) => b.count - a.count),
)

/** The pairs as a symmetric matrix over the columns they name, strongest first. */
const heat = computed(() => {
  const pairs = props.facts.missingness?.pairs ?? []
  if (!pairs.length) return null
  const strength = new Map<string, number>()
  for (const p of pairs)
    for (const name of [p.a, p.b]) strength.set(name, (strength.get(name) ?? 0) + p.jaccard)
  const labels = [...strength]
    .sort((a, b) => b[1] - a[1])
    .slice(0, MAX_HEAT_COLUMNS)
    .map(([name]) => name)
  const at = new Map(labels.map((name, i) => [name, i]))
  const values: number[][] = labels.map((_, y) => labels.map((__, x) => (x === y ? 1 : 0)))
  for (const p of pairs) {
    const x = at.get(p.a)
    const y = at.get(p.b)
    if (x === undefined || y === undefined) continue
    values[y]![x] = p.jaccard
    values[x]![y] = p.jaccard
  }
  return { labels, values }
})
</script>

<style scoped>
.sub {
  font-size: 14px;
  margin: 16px 0 4px;
}
.note {
  margin: 4px 0 0;
}
</style>
