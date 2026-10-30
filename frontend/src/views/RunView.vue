<template>
  <div v-if="overview" class="stack">
    <p v-if="data.error.value" class="error">{{ data.error.value }}</p>
    <MetricStrip :metrics="metrics" />
    <SamplingNotice v-if="header.dataset.sampling" :sampling="header.dataset.sampling" />
    <nav class="links" aria-label="Run links">
      <router-link :to="{ name: 'findings', params: { runId } }">Findings explorer</router-link>
      ·
      <router-link :to="compareTo" data-role="compare-link">{{
        previousId ? 'Compare with previous run' : 'Compare with another run'
      }}</router-link>
    </nav>

    <div class="pair">
      <FindingCounts :counts="overview.finding_counts" :run-id="runId" />
      <ColumnTypes v-if="data.columns.value" :columns="data.columns.value" />
    </div>
    <TopFindings :run-id="runId" />
    <ColumnTable v-if="data.columns.value" :run-id="runId" :columns="data.columns.value" />
    <MissingnessPanel
      v-if="data.columns.value && data.facts.value"
      :columns="data.columns.value"
      :facts="data.facts.value"
    />
    <PolicyVerdict v-if="header.policy" :policy="header.policy" />
    <RunHealthPanel :health="header.health" :notes="header.notes" />
    <SamplePreview :run-id="runId" />
    <DtypeSavings v-if="data.dtypes.value" :dtypes="data.dtypes.value" />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../api/client'
import ColumnTable from '../components/overview/ColumnTable.vue'
import ColumnTypes from '../components/overview/ColumnTypes.vue'
import DtypeSavings from '../components/overview/DtypeSavings.vue'
import MetricStrip, { type Metric } from '../components/overview/MetricStrip.vue'
import MissingnessPanel from '../components/overview/MissingnessPanel.vue'
import SamplePreview from '../components/overview/SamplePreview.vue'
import { formatBytes, missingTone, percent } from '../components/overview/overview'
import FindingCounts from '../components/run/FindingCounts.vue'
import PolicyVerdict from '../components/run/PolicyVerdict.vue'
import RunHealthPanel from '../components/run/RunHealthPanel.vue'
import SamplingNotice from '../components/run/SamplingNotice.vue'
import TopFindings from '../components/run/TopFindings.vue'
import { useOverviewData } from '../composables/useOverviewData'
import { useRun } from '../composables/useRun'
import { formatCount } from '../format'

const props = defineProps<{ runId: string }>()

const { overview } = useRun()
const data = useOverviewData(() => props.runId)
// Only read once `overview` is loaded, by the template's v-if.
const header = computed(() => overview.value!.header)
const metrics = computed<Metric[]>(() => {
  const { dataset, plan } = header.value
  const { facts, columns, duplicates, sampleSize } = data
  const nulls = columns.value?.reduce((sum, c) => sum + c.counts.nulls, 0)
  const cells = dataset.n_rows * dataset.n_columns
  const missing = nulls === undefined || cells === 0 ? null : nulls / cells
  const strip: Metric[] = [
    { label: 'Rows', value: formatCount(dataset.n_rows) },
    { label: 'Columns', value: formatCount(dataset.n_columns) },
  ]
  if (facts.value?.approx_memory_bytes != null)
    strip.push({ label: 'Memory', value: formatBytes(facts.value.approx_memory_bytes) })
  if (missing !== null)
    strip.push({
      label: 'Missing',
      value: percent(missing),
      note: 'of all cells',
      tone: missingTone(missing),
    })
  if (duplicates.value)
    strip.push({
      label: 'Duplicate rows',
      value: formatCount(duplicates.value.duplicate_rows),
      note: percent(duplicates.value.duplicate_share),
      tone: duplicates.value.duplicate_share >= 0.05 ? 'amber' : 'none',
    })
  if (sampleSize.value) {
    const { verdict } = sampleSize.value
    strip.push({
      label: 'Sample adequacy',
      value: verdict,
      tone: verdict === 'adequate' ? 'green' : verdict === 'limited' ? 'amber' : 'red',
    })
  }
  strip.push({ label: 'Depth', value: plan.depth ?? 'custom' })
  if (dataset.sampling)
    strip.push({
      label: 'Sampling',
      value: dataset.sampling.strategy,
      note: `${formatCount(dataset.sampling.sampled_rows)} of ${formatCount(dataset.sampling.original_rows)} rows`,
    })
  return strip
})
// The run before this one in its dataset, which the compare view pre-fills as base.
const previousId = ref('')
const compareTo = computed(() => ({
  name: 'compare',
  query: previousId.value ? { base: previousId.value, head: props.runId } : { head: props.runId },
}))

watch(overview, (loaded) => loaded && findPrevious(loaded.header.dataset.key), { immediate: true })

async function findPrevious(key: string) {
  previousId.value = ''
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
.links {
  margin: 0;
}
.pair {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
}
</style>
