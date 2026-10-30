<template>
  <div v-if="overview" class="stack">
    <section class="card" aria-label="Run">
      <p v-if="header.dataset.source.location" class="muted">
        Source <code>{{ header.dataset.source.location }}</code>
      </p>
      <p>
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
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../api/client'
import FindingCounts from '../components/run/FindingCounts.vue'
import PolicyVerdict from '../components/run/PolicyVerdict.vue'
import RunHealthPanel from '../components/run/RunHealthPanel.vue'
import SamplingNotice from '../components/run/SamplingNotice.vue'
import TopFindings from '../components/run/TopFindings.vue'
import { useRun } from '../composables/useRun'

const props = defineProps<{ runId: string }>()

const { overview } = useRun()
// Only read once `overview` is loaded, by the template's v-if.
const header = computed(() => overview.value!.header)
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
