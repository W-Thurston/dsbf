<template>
  <div class="stack">
    <section class="card" aria-labelledby="compare-title">
      <p class="muted crumbs"><router-link to="/">Datasets</router-link> /</p>
      <h1 id="compare-title" class="card-title">Compare runs</h1>
      <div class="pickers">
        <label>
          Base (compare against)
          <select v-model="base" data-role="base">
            <option value="">Choose a run</option>
            <option v-for="run in runs" :key="run.run_id" :value="run.run_id">
              {{ label(run) }}
            </option>
          </select>
        </label>
        <label>
          Head (describe)
          <select v-model="head" data-role="head">
            <option value="">Choose a run</option>
            <option v-for="run in runs" :key="run.run_id" :value="run.run_id">
              {{ label(run) }}
            </option>
          </select>
        </label>
      </div>
      <p v-if="runsError" class="error">{{ runsError }}</p>
    </section>

    <p v-if="!base || !head" class="muted" data-role="compare-prompt">
      Choose a base and a head run to see what changed.
    </p>
    <p v-else-if="base === head" class="muted">Choose two different runs.</p>
    <p v-else-if="error" class="error">{{ error }}</p>
    <p v-else-if="diff === null" class="loading">Comparing…</p>
    <DiffSections v-else :diff="diff" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, unwrap } from '../api/client'
import DiffSections from '../components/compare/DiffSections.vue'
import { formatDateTime } from '../format'
import type { components } from '../generated/api'

type Summary = components['schemas']['RunSummary']

const route = useRoute()
const router = useRouter()

const runs = ref<Summary[]>([])
const runsError = ref('')
const base = ref(String(route.query.base ?? ''))
const head = ref(String(route.query.head ?? ''))
const diff = ref<components['schemas']['RunDiff'] | null>(null)
const error = ref('')

function label(run: Summary): string {
  return `${run.dataset_name} · ${formatDateTime(run.finished_at)} · ${run.run_id.slice(0, 8)}`
}

async function compare() {
  diff.value = null
  error.value = ''
  if (!base.value || !head.value || base.value === head.value) return
  try {
    diff.value = await unwrap(
      api.GET('/v2/runs/compare', { params: { query: { base: base.value, head: head.value } } }),
    )
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

onMounted(async () => {
  try {
    runs.value = (await unwrap(api.GET('/v2/runs', { params: { query: { limit: 500 } } }))).items
  } catch (caught) {
    runsError.value = caught instanceof Error ? caught.message : String(caught)
  }
  await compare()
})

watch([base, head], () => {
  void router.replace({ query: { base: base.value || undefined, head: head.value || undefined } })
  void compare()
})
</script>

<style scoped>
.crumbs {
  margin: 0 0 4px;
}
.pickers {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}
.pickers label {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
</style>
