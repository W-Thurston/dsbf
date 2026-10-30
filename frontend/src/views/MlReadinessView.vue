<template>
  <div class="stack">
    <section class="card" aria-labelledby="ml-title">
      <h1 id="ml-title" class="run-name">ML readiness</h1>
      <p class="muted">
        Read from other checks' findings. A dimension whose checks could not be read has no level;
        it is not clear, only unassessed.
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
  </div>
</template>

<script setup lang="ts">
import CheckCard from '../components/relationships/CheckCard.vue'
import DimensionCard from '../components/quality/DimensionCard.vue'
import {
  gateCaveat,
  gateClass,
  gateWords,
  type ReadinessOutput,
} from '../components/quality/quality'

defineProps<{ runId: string }>()

const asReadiness = (output: unknown) => output as ReadinessOutput
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
.run-name {
  font-size: 22px;
}
</style>
