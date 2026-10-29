<template>
  <section class="card" aria-labelledby="sample-title">
    <h2 id="sample-title" class="card-title">Sample values</h2>
    <p v-if="missing" class="muted">This run kept no data snapshot, so there is no sample.</p>
    <p v-else-if="error" class="error">{{ error }}</p>
    <p v-else-if="result === null" class="loading">Loading…</p>
    <template v-else>
      <SamplingNotice v-if="result.sampling" :sampling="result.sampling" />
      <table class="data-table">
        <thead>
          <tr>
            <th scope="col">Row</th>
            <th scope="col">{{ column }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, index) in result.rows" :key="result.offset + index">
            <td>{{ result.offset + index + 1 }}</td>
            <td>{{ row[at] === null ? 'null' : row[at] }}</td>
          </tr>
        </tbody>
      </table>
      <PagerNav
        :offset="result.offset"
        :limit="result.limit"
        :total="result.total"
        @update:offset="load"
      />
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { ApiError, api, unwrap } from '../../api/client'
import type { components } from '../../generated/api'
import PagerNav from '../PagerNav.vue'
import SamplingNotice from '../run/SamplingNotice.vue'

const props = defineProps<{ runId: string; column: string }>()

const ROWS = 20
const result = ref<components['schemas']['ArtifactRows'] | null>(null)
const error = ref('')
const missing = ref(false)
const at = computed(() => result.value?.columns.indexOf(props.column) ?? -1)

async function load(offset: number) {
  error.value = ''
  try {
    result.value = await unwrap(
      api.GET('/v2/runs/{run_id}/snapshot', {
        params: { path: { run_id: props.runId }, query: { limit: ROWS, offset } },
      }),
    )
  } catch (caught) {
    if (caught instanceof ApiError && caught.status === 404) missing.value = true
    else error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

watch(
  () => [props.runId, props.column],
  () => load(0),
  { immediate: true },
)
</script>
