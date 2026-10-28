<template>
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
</template>

<script setup lang="ts">
import { api, unwrap } from '../api/client'
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
</script>

<style scoped>
.crumbs {
  margin: 0 0 4px;
}
</style>
