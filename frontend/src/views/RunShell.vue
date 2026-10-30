<template>
  <p v-if="run.error.value" class="error">{{ run.error.value }}</p>
  <p v-else-if="header === null" class="loading">Loading…</p>
  <div v-else class="stack">
    <header class="run-header" aria-label="Run">
      <p class="crumbs" data-role="breadcrumb">
        <router-link to="/">DSBF</router-link> ›
        <router-link
          v-if="header.dataset.key"
          :to="{ name: 'dataset', params: { datasetKey: header.dataset.key } }"
          >{{ header.dataset.name }}</router-link
        ><span v-else>{{ header.dataset.name }}</span> ›
        <span class="run-date">{{ formatDateTime(header.finished_at ?? header.started_at) }}</span>
      </p>
      <p class="muted facts" data-role="run-facts">
        {{ formatCount(header.dataset.n_rows) }} rows × {{ header.dataset.n_columns }} columns ·
        {{ header.plan.depth ?? 'custom' }} depth<template v-if="header.dataset.sampling">
          · <span class="sampled">sampled</span></template
        >
        · dsbf {{ header.plan.dsbf_version }}
      </p>
    </header>
    <HealthBar :dimensions="run.dimensions.value" :tab="tab ?? 'overview'" />
    <RunTabs :run-id="runId" :tab="tab" />
    <router-view />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import HealthBar from '../components/shell/HealthBar.vue'
import RunTabs from '../components/shell/RunTabs.vue'
import type { Tab } from '../components/shell/health'
import { provideRun } from '../composables/useRun'
import { formatCount, formatDateTime } from '../format'

const props = defineProps<{ runId: string }>()

const run = provideRun(() => props.runId)
const route = useRoute()
const header = computed(() => run.overview.value?.header ?? null)
// A page that is no tab (the findings explorer, one column) leaves every tab inactive.
const tab = computed(() => (route.meta.tab as Tab | undefined) ?? null)
</script>

<style scoped>
.crumbs {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}
.facts {
  margin: 2px 0 0;
}
.sampled {
  color: var(--warn-fg);
  background: var(--warn-bg);
  padding: 0 6px;
  border-radius: 8px;
}
</style>
