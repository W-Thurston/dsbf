<template>
  <div class="stack">
    <section class="card" aria-labelledby="quality-title">
      <h1 id="quality-title" class="run-name">Quality</h1>
      <p class="muted">
        The summary is read from other checks' findings. A dimension whose checks could not be read
        has no level; it is not clear, only unassessed.
      </p>
    </section>

    <CheckCard
      v-slot="{ output }"
      :run-id="runId"
      check-id="quality_dimensions"
      title="Quality dimensions"
    >
      <template v-for="q in [asQuality(output)]" :key="q.total_columns">
        <div class="grid">
          <DimensionCard
            v-for="(d, key) in q.dimensions"
            :id="`quality-${key}`"
            :key="key"
            :run-id="runId"
            :label="String(key)"
            :assessed="d.assessed"
            :level="d.level"
            :coverage="d.coverage"
            :inputs="d.inputs"
            :findings="d.findings"
          />
        </div>
        <h3 class="group">Clean columns</h3>
        <p v-if="!q.clean_columns.length" class="muted">Every column is flagged by a dimension.</p>
        <ul v-else class="clean" data-role="clean-columns">
          <li v-for="name in q.clean_columns" :key="name">{{ name }}</li>
        </ul>
        <p class="muted">
          {{ q.clean_columns.length }} of {{ q.total_columns }} columns. A column no check could
          read is here too; see the coverage below.
        </p>
      </template>
    </CheckCard>

    <ColumnCoverage :run-id="runId" />
  </div>
</template>

<script setup lang="ts">
import CheckCard from '../components/relationships/CheckCard.vue'
import ColumnCoverage from '../components/quality/ColumnCoverage.vue'
import DimensionCard from '../components/quality/DimensionCard.vue'
import { type QualityOutput } from '../components/quality/quality'

defineProps<{ runId: string }>()

const asQuality = (output: unknown) => output as QualityOutput
</script>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.clean {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  padding-left: 18px;
}
</style>
