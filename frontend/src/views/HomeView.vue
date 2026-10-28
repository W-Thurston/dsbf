<template>
  <div class="stack">
    <section class="card" aria-labelledby="datasets-title">
      <h1 id="datasets-title" class="card-title">Datasets</h1>
      <p v-if="datasets.error.value" class="error">{{ datasets.error.value }}</p>
      <p v-else-if="datasets.page.value === null" class="loading">Loading…</p>
      <p v-else-if="!datasets.page.value.items.length" class="muted">
        No runs are stored yet. Profile a file with <code>dsbf profile data.csv -s store</code>.
      </p>
      <template v-else>
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">Dataset</th>
              <th scope="col">Last run</th>
              <th scope="col" class="num">Runs</th>
              <th scope="col">Highest</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="dataset in datasets.page.value.items" :key="dataset.key">
              <td>
                <router-link :to="{ name: 'dataset', params: { datasetKey: dataset.key } }">{{
                  dataset.name
                }}</router-link>
              </td>
              <td>
                <router-link :to="{ name: 'run', params: { runId: dataset.last_run_id } }">{{
                  formatDateTime(dataset.last_run_at)
                }}</router-link>
              </td>
              <td class="num">{{ dataset.n_runs }}</td>
              <td><SeverityBadge :severity="dataset.last_max_severity" /></td>
            </tr>
          </tbody>
        </table>
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
import { api, unwrap } from '../api/client'
import PagerNav from '../components/PagerNav.vue'
import RunTable from '../components/RunTable.vue'
import SeverityBadge from '../components/SeverityBadge.vue'
import { usePage } from '../composables/usePage'
import { formatDateTime } from '../format'

const datasets = usePage((limit, offset) =>
  unwrap(api.GET('/v2/datasets', { params: { query: { limit, offset } } })),
)
const runs = usePage((limit, offset) =>
  unwrap(api.GET('/v2/runs', { params: { query: { limit, offset } } })),
)
</script>
