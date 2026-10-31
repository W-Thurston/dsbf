<template>
  <div class="stack">
    <section aria-labelledby="datasets-title">
      <h1 id="datasets-title" class="card-title">Datasets</h1>
      <p v-if="datasets.error.value" class="error">{{ datasets.error.value }}</p>
      <p v-else-if="datasets.page.value === null" class="loading">Loading…</p>
      <p v-else-if="!datasets.page.value.items.length" class="muted">
        No runs are stored yet. Profile a file with <code>dsbf profile data.csv -s store</code>.
      </p>
      <template v-else>
        <label class="search">
          <span class="visually-hidden">Search datasets</span>
          <input v-model="query" type="search" placeholder="Search datasets" data-role="search" />
        </label>
        <p v-if="!shown.length" class="muted" data-role="no-match">
          No dataset on this page matches “{{ query }}”.
        </p>
        <DatasetCards v-else :datasets="shown" />
        <PagerNav
          v-model:offset="datasets.offset.value"
          :limit="datasets.size"
          :total="datasets.page.value.total"
        />
      </template>
    </section>

    <section class="card" aria-labelledby="runs-title">
      <h2 id="runs-title" class="card-title">Recent runs</h2>
      <p v-if="runs.error.value" class="error">{{ runs.error.value }}</p>
      <p v-else-if="runs.page.value === null" class="loading">Loading…</p>
      <p v-else-if="!runs.page.value.items.length" class="muted">No runs.</p>
      <template v-else>
        <RunTable :runs="runs.page.value.items" />
        <PagerNav
          v-model:offset="runs.offset.value"
          :limit="runs.size"
          :total="runs.page.value.total"
        />
      </template>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { api, unwrap } from '../api/client'
import DatasetCards from '../components/dataset/DatasetCards.vue'
import PagerNav from '../components/PagerNav.vue'
import RunTable from '../components/RunTable.vue'
import { usePage } from '../composables/usePage'

const datasets = usePage((limit, offset) =>
  unwrap(api.GET('/v2/datasets', { params: { query: { limit, offset } } })),
)
const query = ref('')
// The search narrows the datasets on the current page by name or key.
const shown = computed(() => {
  const needle = query.value.trim().toLowerCase()
  const items = datasets.page.value?.items ?? []
  return needle ? items.filter((d) => `${d.name} ${d.key}`.toLowerCase().includes(needle)) : items
})
const runs = usePage((limit, offset) =>
  unwrap(api.GET('/v2/runs', { params: { query: { limit, offset } } })),
)
</script>

<style scoped>
.search input {
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: inherit;
  font: inherit;
  width: 100%;
  max-width: 360px;
  margin-bottom: 16px;
}
</style>
