<template>
  <div class="stack">
    <section class="card" aria-labelledby="explorer-title">
      <p class="muted crumbs">
        <router-link to="/">Datasets</router-link> /
        <router-link :to="{ name: 'run', params: { runId } }">Run</router-link> /
      </p>
      <h1 id="explorer-title" class="run-name">Findings</h1>
      <FindingFilters :query="query" :dimensions="dimensions" @change="go" />
    </section>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="page === null" class="loading">Loading…</p>
    <section v-else class="card" aria-label="Findings list">
      <p class="muted" role="status">{{ formatCount(page.total) }} findings</p>
      <ol v-if="page.items.length" class="findings">
        <li v-for="item in page.items" :key="item.id" :class="{ open: item.id === query.finding }">
          <SeverityBadge :severity="item.severity" />
          <router-link :to="{ query: toLocationQuery({ ...query, finding: item.id }) }">
            <strong>{{ item.title }}</strong>
          </router-link>
          <span class="muted"> · {{ item.dimension }} · {{ item.code }}</span>
          <span v-if="item.subject.columns.length" class="muted">
            · {{ item.subject.columns.join(' and ') }}</span
          >
        </li>
      </ol>
      <p v-else class="muted">No findings match.</p>
      <PagerNav
        :offset="page.offset"
        :limit="page.limit"
        :total="page.total"
        @update:offset="(offset) => go({ ...query, offset })"
      />
    </section>

    <p v-if="detailError" class="error">{{ detailError }}</p>
    <FindingDetail
      v-else-if="detail"
      :run-id="runId"
      :finding="detail"
      @close="go({ ...query, finding: '' })"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, unwrap } from '../api/client'
import FindingDetail from '../components/findings/FindingDetail.vue'
import FindingFilters from '../components/findings/FindingFilters.vue'
import PagerNav from '../components/PagerNav.vue'
import SeverityBadge from '../components/SeverityBadge.vue'
import {
  apiQuery,
  parseFindingQuery,
  toLocationQuery,
  type FindingQuery,
} from '../composables/findingQuery'
import { formatCount } from '../format'
import type { components } from '../generated/api'

const props = defineProps<{ runId: string }>()

const route = useRoute()
const router = useRouter()
const query = computed(() => parseFindingQuery(route.query))

const page = ref<components['schemas']['Page_Finding_'] | null>(null)
const detail = ref<components['schemas']['Finding'] | null>(null)
const dimensions = ref<string[]>([])
const error = ref('')
const detailError = ref('')

function go(next: FindingQuery) {
  void router.push({ query: toLocationQuery(next) })
}

function message(caught: unknown): string {
  return caught instanceof Error ? caught.message : String(caught)
}

async function loadPage() {
  error.value = ''
  try {
    page.value = await unwrap(
      api.GET('/v2/runs/{run_id}/findings', {
        params: { path: { run_id: props.runId }, query: apiQuery(query.value) },
      }),
    )
  } catch (caught) {
    error.value = message(caught)
  }
}

async function loadDetail() {
  detail.value = null
  detailError.value = ''
  if (!query.value.finding) return
  try {
    detail.value = await unwrap(
      api.GET('/v2/runs/{run_id}/findings/{finding_id}', {
        params: { path: { run_id: props.runId, finding_id: query.value.finding } },
      }),
    )
  } catch (caught) {
    detailError.value = message(caught)
  }
}

// The list depends on every filter; the open finding only on its id.
watch(() => JSON.stringify({ ...query.value, finding: '' }), loadPage, { immediate: true })
watch(() => query.value.finding, loadDetail, { immediate: true })

unwrap(api.GET('/v2/runs/{run_id}', { params: { path: { run_id: props.runId } } }))
  .then((overview) => {
    dimensions.value = [...new Set(overview.finding_counts.map((c) => c.dimension))].sort()
  })
  .catch(() => {
    // The filter still works typed; only its choices are missing.
  })
</script>

<style scoped>
.run-name {
  font-size: 22px;
}
.crumbs {
  margin: 0 0 4px;
}
.findings {
  list-style: none;
  padding: 0;
  margin: 0;
}
.findings li {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}
.findings li.open {
  background: var(--surface-2, transparent);
}
</style>
