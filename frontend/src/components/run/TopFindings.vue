<template>
  <section class="card" aria-labelledby="top-title">
    <h2 id="top-title" class="card-title">Top findings</h2>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="page === null" class="loading">Loading…</p>
    <p v-else-if="!page.items.length" class="muted">No findings.</p>
    <ol v-else class="findings">
      <li v-for="finding in page.items" :key="finding.id">
        <SeverityBadge :severity="finding.severity" />
        <router-link :to="{ name: 'findings', params: { runId }, query: { finding: finding.id } }"
          ><strong>{{ finding.title }}</strong></router-link
        >
        <span class="muted"> · {{ finding.dimension }}</span>
        <p>{{ finding.message }}</p>
      </li>
    </ol>
    <p v-if="page && page.total > page.items.length" class="muted">
      Showing {{ page.items.length }} of {{ formatCount(page.total) }}.
    </p>
    <p v-if="page && page.total">
      <router-link :to="{ name: 'findings', params: { runId } }">Explore all findings</router-link>
    </p>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, unwrap } from '../../api/client'
import { formatCount } from '../../format'
import type { components } from '../../generated/api'
import SeverityBadge from '../SeverityBadge.vue'

const props = defineProps<{ runId: string }>()

const TOP = 10
const page = ref<components['schemas']['Page_Finding_'] | null>(null)
const error = ref('')

onMounted(async () => {
  try {
    page.value = await unwrap(
      api.GET('/v2/runs/{run_id}/findings', {
        params: { path: { run_id: props.runId }, query: { limit: TOP } },
      }),
    )
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
})
</script>

<style scoped>
.findings {
  list-style: none;
  padding: 0;
  margin: 0;
}
.findings li {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}
.findings p {
  margin: 2px 0 0;
}
</style>
