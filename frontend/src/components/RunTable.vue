<template>
  <table class="data-table">
    <thead>
      <tr>
        <th scope="col">Dataset</th>
        <th scope="col">Finished</th>
        <th scope="col" class="num">Rows</th>
        <th scope="col" class="num">Columns</th>
        <th scope="col" class="num">Findings</th>
        <th scope="col">Highest</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="run in runs" :key="run.run_id">
        <td>
          <router-link :to="{ name: 'run', params: { runId: run.run_id } }">{{
            run.dataset_name
          }}</router-link>
        </td>
        <td>{{ formatDateTime(run.finished_at) }}</td>
        <td class="num">{{ formatCount(run.n_rows) }}</td>
        <td class="num">{{ formatCount(run.n_columns) }}</td>
        <td class="num">{{ formatCount(run.n_findings) }}</td>
        <td><SeverityBadge :severity="run.max_severity" /></td>
      </tr>
    </tbody>
  </table>
</template>

<script setup lang="ts">
import { formatCount, formatDateTime } from '../format'
import type { components } from '../generated/api'
import SeverityBadge from './SeverityBadge.vue'

defineProps<{ runs: components['schemas']['RunSummary'][] }>()
</script>
