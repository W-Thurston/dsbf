<template>
  <div class="stack">
    <section class="card" aria-labelledby="quality-title">
      <p class="muted crumbs">
        <router-link to="/">Datasets</router-link> /
        <router-link :to="{ name: 'run', params: { runId } }">Run</router-link> /
      </p>
      <h1 id="quality-title" class="run-name">Quality and ML readiness</h1>
      <p class="muted">
        Both summaries are read from other checks' findings. A dimension whose checks could not be
        read has no level; it is not clear, only unassessed.
      </p>
    </section>

    <CheckCard v-slot="{ output }" :run-id="runId" check-id="ml_readiness" title="ML readiness">
      <template v-for="ml in [asReadiness(output)]" :key="ml.gate">
        <p class="gate" data-role="gate">
          <span :class="['badge', gateClass(ml.gate)]">{{ gateWords(ml.gate) }}</span>
          <span v-if="gateCaveat(ml, labels(ml))" class="muted" data-role="caveat">{{
            gateCaveat(ml, labels(ml))
          }}</span>
        </p>
        <div class="grid">
          <DimensionCard
            v-for="(d, key) in ml.dimensions"
            :id="`ml-${key}`"
            :key="key"
            :run-id="runId"
            :label="d.label"
            :assessed="d.assessed"
            :level="d.level"
            :coverage="d.coverage"
            :inputs="d.inputs"
            :findings="d.findings"
          />
        </div>
        <p class="muted" data-role="column-status">
          {{ ml.action_columns.length }} column(s) need action,
          {{ ml.advisory_columns.length }} have advice only, {{ ml.clean_columns.length }} have no
          finding.
        </p>
      </template>
    </CheckCard>

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
import {
  gateCaveat,
  gateClass,
  gateWords,
  type QualityOutput,
  type ReadinessOutput,
} from '../components/quality/quality'

defineProps<{ runId: string }>()

const asReadiness = (output: unknown) => output as ReadinessOutput
const asQuality = (output: unknown) => output as QualityOutput
const labels = (ml: ReadinessOutput) =>
  Object.fromEntries(Object.entries(ml.dimensions).map(([key, d]) => [key, d.label]))
</script>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.gate {
  display: flex;
  gap: 12px;
  align-items: center;
  font-size: 16px;
  margin-bottom: 12px;
}
.clean {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  padding-left: 18px;
}
</style>
