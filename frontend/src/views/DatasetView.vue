<template>
  <div class="stack">
    <section class="card" aria-labelledby="dataset-title">
      <p class="muted crumbs"><router-link to="/">Datasets</router-link> /</p>
      <h1 id="dataset-title" class="card-title">Runs of {{ datasetKey }}</h1>
      <p v-if="runs.error.value" class="error">{{ runs.error.value }}</p>
      <p v-else-if="runs.page.value === null" class="loading">Loading…</p>
      <template v-else>
        <RunTable :runs="runs.page.value.items" />
        <PagerNav
          v-model:offset="runs.offset.value"
          :limit="runs.size"
          :total="runs.page.value.total"
        />
      </template>
    </section>

    <section class="card" aria-labelledby="lineage-title">
      <h2 id="lineage-title" class="card-title">Lineage</h2>
      <p v-if="lineage.error.value" class="error">{{ lineage.error.value }}</p>
      <p v-else-if="lineage.page.value === null" class="loading">Loading…</p>
      <template v-else>
        <LineageTable :entries="lineage.page.value.items" />
        <PagerNav
          v-model:offset="lineage.offset.value"
          :limit="lineage.size"
          :total="lineage.page.value.total"
        />
      </template>
    </section>

    <DatasetSettings v-if="runs.page.value?.items.length" :dataset-key="datasetKey" />

    <FindingHistory
      v-if="runs.page.value?.items.length"
      :dataset-key="datasetKey"
      :latest-run-id="runs.page.value.items[0]!.run_id"
    />
  </div>
</template>

<script setup lang="ts">
import { api, unwrap } from '../api/client'
import FindingHistory from '../components/dataset/FindingHistory.vue'
import DatasetSettings from '../components/settings/DatasetSettings.vue'
import LineageTable from '../components/dataset/LineageTable.vue'
import PagerNav from '../components/PagerNav.vue'
import RunTable from '../components/RunTable.vue'
import { usePage } from '../composables/usePage'

const props = defineProps<{ datasetKey: string }>()

const runs = usePage((limit, offset) =>
  unwrap(
    api.GET('/v2/datasets/{key}/runs', {
      params: { path: { key: props.datasetKey }, query: { limit, offset } },
    }),
  ),
)

const lineage = usePage((limit, offset) =>
  unwrap(
    api.GET('/v2/datasets/{key}/lineage', {
      params: { path: { key: props.datasetKey }, query: { limit, offset } },
    }),
  ),
)
</script>

<style scoped>
.crumbs {
  margin: 0 0 4px;
}
</style>
