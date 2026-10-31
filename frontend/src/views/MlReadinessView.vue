<template>
  <div class="stack">
    <h1 class="visually-hidden">ML readiness</h1>
    <p v-if="state === 'loading'" class="loading">Loading…</p>
    <section v-else-if="!ml" class="card" aria-labelledby="ml-title">
      <h2 id="ml-title" class="card-title">ML readiness</h2>
      <p class="muted">
        ML readiness did not run in this run{{ reason ? `: ${reason}` : '' }}. It is read from other
        checks' findings.
      </p>
    </section>
    <template v-else>
      <LevelBanner
        :mark="gateMark"
        :headline="gateWords(ml.gate)"
        :detail="gateCaveat(ml, labels)"
        data-role="gate"
      />
      <p class="muted" data-role="column-status">
        {{ ml.action_columns.length }} column(s) need action, {{ ml.advisory_columns.length }} have
        advice only, {{ ml.clean_columns.length }} have no finding. A dimension whose checks could
        not be read has no level; it is not clear, only unassessed.
      </p>
      <div class="grid">
        <TrafficCard
          v-for="(d, key) in ml.dimensions"
          :id="`ml-${key}`"
          :key="key"
          :title="d.label"
          :mark="d.level && d.assessed ? d.level : 'none'"
          :summary="summary(d)"
          :data-dimension="`ml-${key}`"
          class="jump"
          role="button"
          tabindex="0"
          @click="show(String(key))"
          @keydown.enter="show(String(key))"
        >
          <span :class="['badge', levelClass(d.assessed ? d.level : null)]" data-role="level">{{
            levelWords(d.assessed ? d.level : null)
          }}</span>
        </TrafficCard>
      </div>

      <DimensionSection
        v-for="(d, key) in ml.dimensions"
        :key="key"
        :run-id="runId"
        :name="String(key)"
        :label="d.label"
        :dimension="d"
        id-prefix="ml"
      />

      <TransformPreview :run-id="runId" />

      <section class="card" aria-labelledby="ready-title">
        <h2 id="ready-title" class="card-title">Ready features</h2>
        <p v-if="!ml.clean_columns.length" class="muted">Every column has a finding.</p>
        <ul v-else class="clean" data-role="ready-features">
          <li v-for="name in ml.clean_columns" :key="name">{{ name }}</li>
        </ul>
        <p class="muted">
          {{ ml.clean_columns.length }} of {{ ml.total_columns }} columns have no readiness finding.
        </p>
      </section>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../api/client'
import LevelBanner from '../components/charts/LevelBanner.vue'
import TrafficCard from '../components/charts/TrafficCard.vue'
import DimensionSection from '../components/quality/DimensionSection.vue'
import {
  gateCaveat,
  gateWords,
  levelClass,
  levelWords,
  percent,
  unassessedReason,
  type ReadinessDimension,
  type ReadinessOutput,
} from '../components/quality/quality'
import TransformPreview from '../components/readiness/TransformPreview.vue'
import type { Mark } from '../components/shell/health'

const props = defineProps<{ runId: string }>()

const ml = ref<ReadinessOutput | null>(null)
const state = ref<'loading' | 'ready'>('loading')
const reason = ref('')

watch(
  () => props.runId,
  async (run) => {
    state.value = 'loading'
    ml.value = null
    reason.value = ''
    try {
      const result = await unwrap(
        api.GET('/v2/runs/{run_id}/checks/{check_id}', {
          params: { path: { run_id: run, check_id: 'ml_readiness' } },
        }),
      )
      if (result.status === 'ok') ml.value = result.output as unknown as ReadinessOutput
      else reason.value = result.notes[0]?.message ?? `status ${result.status}`
    } catch {
      ml.value = null
    }
    state.value = 'ready'
  },
  { immediate: true },
)

const GATE_MARK: Record<ReadinessOutput['gate'], Mark> = {
  ready: 'green',
  needs_work: 'amber',
  not_ready: 'red',
  not_assessed: 'none',
}
const gateMark = computed(() => GATE_MARK[ml.value?.gate ?? 'not_assessed'])
const labels = computed(() =>
  Object.fromEntries(Object.entries(ml.value?.dimensions ?? {}).map(([k, d]) => [k, d.label])),
)

function summary(d: ReadinessDimension): string {
  if (!d.assessed) return unassessedReason(d.inputs)
  if (!d.findings.length) return 'No findings.'
  const columns = d.action_columns.length + d.advisory_columns.length
  return `${d.findings.length} ${d.findings.length === 1 ? 'finding' : 'findings'}, ${columns} ${columns === 1 ? 'column' : 'columns'}; checks read ${percent(d.coverage)}.`
}

// A card opens its section and brings it into view.
function show(key: string) {
  const section = document.getElementById(`ml-section-${key}`) as HTMLDetailsElement | null
  if (!section) return
  section.open = true
  section.scrollIntoView({ behavior: 'smooth', block: 'start' })
}
</script>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 12px;
}
.jump {
  cursor: pointer;
}
.clean {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  list-style: none;
  padding: 0;
}
.clean li {
  padding: 1px 10px;
  border: 1px solid var(--border);
  border-radius: 10px;
}
</style>
