<template>
  <table class="data-table" data-section="lineage">
    <thead>
      <tr>
        <th scope="col">Run</th>
        <th scope="col">Source</th>
        <th scope="col" class="num">Rows</th>
        <th scope="col">Changes</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="entry in entries" :key="entry.run_id" :data-run="entry.run_id">
        <td>
          <router-link :to="{ name: 'run', params: { runId: entry.run_id } }">{{
            formatDateTime(entry.started_at)
          }}</router-link>
        </td>
        <td>
          <code v-if="entry.location">{{ entry.location }}</code>
          <span v-else class="muted">in memory</span>
        </td>
        <td class="num">{{ formatCount(entry.n_rows) }}</td>
        <td>
          <span v-if="entry.location_changed" class="badge" data-flag="location">
            location changed
          </span>
          <span v-if="entry.fingerprint_changed" class="badge" data-flag="fingerprint">
            content changed
          </span>
          <span v-if="!entry.location_changed && !entry.fingerprint_changed" class="muted">
            –
          </span>
        </td>
      </tr>
    </tbody>
  </table>
</template>

<script setup lang="ts">
import { formatCount, formatDateTime } from '../../format'
import type { components } from '../../generated/api'

defineProps<{ entries: components['schemas']['LineageEntry'][] }>()
</script>
