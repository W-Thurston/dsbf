<template>
  <div class="stack">
    <h1 class="visually-hidden">Quality</h1>
    <p v-if="state === 'loading'" class="loading">Loading…</p>
    <section v-else-if="!quality" class="card" aria-labelledby="quality-title">
      <h2 id="quality-title" class="card-title">Quality</h2>
      <p class="muted">
        The quality dimensions did not run in this run{{ reason ? `: ${reason}` : '' }}.
      </p>
    </section>
    <template v-else>
      <LevelBanner
        :mark="trust.level ?? 'none'"
        :headline="trust.headline"
        :detail="trust.detail"
        data-role="trust"
      />
      <div class="grid">
        <TrafficCard
          v-for="(d, key) in quality.dimensions"
          :id="`quality-${key}`"
          :key="key"
          :title="labelOf(String(key))"
          :mark="d.level && d.assessed ? d.level : 'none'"
          :summary="dimensionSummary(d, quality.total_columns)"
          :data-dimension="`quality-${key}`"
          class="jump"
          role="button"
          tabindex="0"
          @click="show(String(key))"
          @keydown.enter="show(String(key))"
        >
          <span :class="['badge', levelClass(d.assessed ? d.level : null)]" data-role="level">{{
            levelWords(d.assessed ? d.level : null)
          }}</span>
          <ul v-if="d.findings.length" class="chips">
            <li v-for="chip in issueChips(d.findings)" :key="chip.code" class="chip">
              {{ words(chip.code) }}<template v-if="chip.count > 1"> ×{{ chip.count }}</template>
            </li>
          </ul>
        </TrafficCard>
      </div>

      <DimensionSection
        v-for="(d, key) in quality.dimensions"
        :key="key"
        :run-id="runId"
        :name="String(key)"
        :label="labelOf(String(key))"
        :dimension="d"
      />

      <section class="card" aria-labelledby="clean-title">
        <h2 id="clean-title" class="card-title">Clean columns</h2>
        <p v-if="!quality.clean_columns.length" class="muted">
          Every column is flagged by a dimension.
        </p>
        <ul v-else class="clean" data-role="clean-columns">
          <li v-for="name in quality.clean_columns" :key="name">{{ name }}</li>
        </ul>
        <p class="muted">
          {{ quality.clean_columns.length }} of {{ quality.total_columns }} columns. A column no
          check could read is here too; see the coverage below.
        </p>
      </section>

      <MissingnessMechanism :run-id="runId" />
      <FuzzyDuplicates :run-id="runId" />
      <ColumnCoverage :run-id="runId" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../api/client'
import LevelBanner from '../components/charts/LevelBanner.vue'
import TrafficCard from '../components/charts/TrafficCard.vue'
import ColumnCoverage from '../components/quality/ColumnCoverage.vue'
import DimensionSection from '../components/quality/DimensionSection.vue'
import FuzzyDuplicates from '../components/quality/FuzzyDuplicates.vue'
import MissingnessMechanism from '../components/quality/MissingnessMechanism.vue'
import {
  dimensionSummary,
  issueChips,
  levelClass,
  levelWords,
  trustOf,
  type QualityOutput,
} from '../components/quality/quality'
import { DIMENSIONS } from '../components/shell/health'
import { words } from '../format'

const props = defineProps<{ runId: string }>()

const quality = ref<QualityOutput | null>(null)
const state = ref<'loading' | 'ready'>('loading')
const reason = ref('')

watch(
  () => props.runId,
  async (run) => {
    state.value = 'loading'
    quality.value = null
    reason.value = ''
    try {
      const result = await unwrap(
        api.GET('/v2/runs/{run_id}/checks/{check_id}', {
          params: { path: { run_id: run, check_id: 'quality_dimensions' } },
        }),
      )
      if (result.status === 'ok') quality.value = result.output as unknown as QualityOutput
      else reason.value = result.notes[0]?.message ?? `status ${result.status}`
    } catch {
      quality.value = null
    }
    state.value = 'ready'
  },
  { immediate: true },
)

const trust = computed(() => trustOf(quality.value as QualityOutput))
const labelOf = (key: string) => DIMENSIONS.find((d) => d.name === key)?.label ?? key

// A card opens its section and brings it into view.
function show(key: string) {
  const section = document.getElementById(`quality-section-${key}`) as HTMLDetailsElement | null
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
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  list-style: none;
  padding: 0;
  margin: 8px 0 0;
}
.chip {
  font-size: 12px;
  padding: 1px 8px;
  border-radius: 10px;
  border: 1px solid var(--border);
  color: var(--text-muted);
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
