<template>
  <section class="card" aria-labelledby="history-title" data-section="history">
    <h2 id="history-title" class="card-title">Finding history</h2>
    <p v-if="!candidates.length" class="muted">The latest run has no findings to follow.</p>
    <template v-else>
      <label>
        Finding in the latest run
        <select v-model="selected" data-role="history-finding">
          <option v-for="finding in candidates" :key="finding.id" :value="finding.id">
            {{ finding.title }} ({{ finding.code }})
          </option>
        </select>
      </label>
      <p v-if="error" class="error">{{ error }}</p>
      <p v-else-if="occurrences === null" class="loading">Loading…</p>
      <table v-else class="data-table">
        <thead>
          <tr>
            <th scope="col">Run</th>
            <th scope="col">Severity</th>
            <th scope="col">Message</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="occurrence in occurrences" :key="occurrence.run_id">
            <td>
              <router-link :to="{ name: 'run', params: { runId: occurrence.run_id } }">{{
                formatDateTime(occurrence.started_at)
              }}</router-link>
            </td>
            <td><SeverityBadge :severity="occurrence.finding.severity" /></td>
            <td>{{ occurrence.finding.message }}</td>
          </tr>
        </tbody>
      </table>
      <p v-if="occurrences" class="muted">
        Appears in {{ plural(occurrences.length, 'run') }}; runs without it are not listed.
      </p>
    </template>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { api, unwrap } from '../../api/client'
import { formatDateTime, plural } from '../../format'
import type { components } from '../../generated/api'
import SeverityBadge from '../SeverityBadge.vue'

const props = defineProps<{ datasetKey: string; latestRunId: string }>()

const TOP = 50
const candidates = ref<components['schemas']['Finding'][]>([])
const selected = ref('')
const occurrences = ref<components['schemas']['FindingOccurrence'][] | null>(null)
const error = ref('')

async function load() {
  error.value = ''
  occurrences.value = null
  try {
    const page = await unwrap(
      api.GET('/v2/datasets/{key}/history', {
        params: { path: { key: props.datasetKey }, query: { finding_id: selected.value } },
      }),
    )
    occurrences.value = page.items
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

onMounted(async () => {
  try {
    const page = await unwrap(
      api.GET('/v2/runs/{run_id}/findings', {
        params: { path: { run_id: props.latestRunId }, query: { limit: TOP } },
      }),
    )
    candidates.value = page.items
    selected.value = page.items[0]?.id ?? ''
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
})

watch(selected, (id) => {
  if (id) void load()
})
</script>
