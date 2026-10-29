<template>
  <p v-if="error" class="error">{{ error }}</p>
  <p v-else-if="overview === null" class="loading">Loading…</p>
  <div v-else class="stack">
    <section class="card" aria-labelledby="run-title">
      <p class="muted crumbs"><router-link to="/">Datasets</router-link> /</p>
      <h1 id="run-title" class="run-name">{{ header.dataset.name }}</h1>
      <p class="muted">
        {{ formatCount(header.dataset.n_rows) }} rows × {{ header.dataset.n_columns }} columns ·
        {{ header.plan.depth ?? 'custom' }} depth ·
        {{ plural(header.plan.checks.length, 'check') }} · finished
        {{ formatDateTime(header.finished_at) }} · dsbf {{ header.plan.dsbf_version }}
      </p>
      <p v-if="header.dataset.source.location" class="muted">
        Source <code>{{ header.dataset.source.location }}</code>
      </p>
      <p>
        <router-link :to="{ name: 'columns', params: { runId } }">Columns</router-link> ·
        <router-link :to="{ name: 'relationships', params: { runId } }">Relationships</router-link>
        · <router-link :to="{ name: 'quality', params: { runId } }">Quality</router-link> ·
        <router-link :to="{ name: 'time-series', params: { runId } }">Time series</router-link> ·
        <router-link :to="{ name: 'findings', params: { runId } }">Findings</router-link> ·
        <router-link :to="compareTo" data-role="compare-link">{{
          previousId ? 'Compare with previous run' : 'Compare with another run'
        }}</router-link>
      </p>
      <SamplingNotice v-if="header.dataset.sampling" :sampling="header.dataset.sampling" />
    </section>

    <PolicyVerdict v-if="header.policy" :policy="header.policy" />
    <RunHealthPanel :health="header.health" :notes="header.notes" />
    <FindingCounts :counts="overview.finding_counts" />
    <TopFindings :run-id="runId" />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, unwrap } from '../api/client'
import FindingCounts from '../components/run/FindingCounts.vue'
import PolicyVerdict from '../components/run/PolicyVerdict.vue'
import RunHealthPanel from '../components/run/RunHealthPanel.vue'
import SamplingNotice from '../components/run/SamplingNotice.vue'
import TopFindings from '../components/run/TopFindings.vue'
import { formatCount, formatDateTime, plural } from '../format'
import type { components } from '../generated/api'

const props = defineProps<{ runId: string }>()

const overview = ref<components['schemas']['RunOverview'] | null>(null)
const error = ref('')
// Only read once `overview` is loaded, by the template's v-else branch.
const header = computed(() => overview.value!.header)
// The run before this one in its dataset, which the compare view pre-fills as base.
const previousId = ref('')
const compareTo = computed(() => ({
  name: 'compare',
  query: previousId.value ? { base: previousId.value, head: props.runId } : { head: props.runId },
}))

onMounted(async () => {
  try {
    overview.value = await unwrap(
      api.GET('/v2/runs/{run_id}', { params: { path: { run_id: props.runId } } }),
    )
    await findPrevious(overview.value.header.dataset.key)
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
})

async function findPrevious(key: string) {
  if (!key) return
  try {
    const runs = await unwrap(
      api.GET('/v2/datasets/{key}/runs', { params: { path: { key }, query: { limit: 500 } } }),
    )
    const at = runs.items.findIndex((run) => run.run_id === props.runId)
    previousId.value = at >= 0 ? (runs.items[at + 1]?.run_id ?? '') : ''
  } catch {
    // The link then offers a head-only pre-fill; the compare view reports real errors.
  }
}
</script>

<style scoped>
.run-name {
  font-size: 22px;
}
.crumbs {
  margin: 0 0 4px;
}
</style>
