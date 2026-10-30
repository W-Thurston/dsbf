<template>
  <div class="stack">
    <section class="card" aria-labelledby="columns-title">
      <h1 id="columns-title" class="run-name">Columns</h1>
      <form class="filters" aria-label="Filter columns" @submit.prevent>
        <label>Name <input v-model.trim="name" type="search" /></label>
        <label>
          Type
          <select v-model="type">
            <option value="">any</option>
            <option v-for="kind in types" :key="kind" :value="kind">{{ kind }}</option>
          </select>
        </label>
      </form>
    </section>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="columns === null" class="loading">Loading…</p>
    <section v-else class="card" aria-label="Column list">
      <p class="muted" role="status">{{ plural(shown.length, 'column') }}</p>
      <table v-if="shown.length" class="data-table">
        <thead>
          <tr>
            <th scope="col">Column</th>
            <th scope="col">Type</th>
            <th scope="col">Null</th>
            <th scope="col">Distinct</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="column in shown" :key="column.name">
            <td>
              <router-link :to="{ name: 'column', params: { runId, column: column.name } }">{{
                column.name
              }}</router-link>
            </td>
            <td>{{ column.semantic_type }}</td>
            <td>{{ formatCount(column.counts.nulls) }}</td>
            <td>{{ formatCount(column.counts.distinct) }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="muted">No columns match.</p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api, unwrap } from '../api/client'
import { formatCount, plural } from '../format'
import type { components } from '../generated/api'

const props = defineProps<{ runId: string }>()

const route = useRoute()
const router = useRouter()
const columns = ref<components['schemas']['ColumnSummary'][] | null>(null)
const error = ref('')

// The filters live in the URL, so a filtered list can be shared and survives a reload.
function param(key: 'q' | 'type') {
  return computed({
    get: () => String(route.query[key] ?? ''),
    set: (value: string) => {
      const query = { ...route.query, [key]: value || undefined }
      void router.replace({ query })
    },
  })
}
const name = param('q')
const type = param('type')

const types = computed(() => [...new Set((columns.value ?? []).map((c) => c.semantic_type))].sort())
const shown = computed(() =>
  (columns.value ?? []).filter(
    (c) =>
      c.name.toLowerCase().includes(name.value.toLowerCase()) &&
      (!type.value || c.semantic_type === type.value),
  ),
)

unwrap(api.GET('/v2/runs/{run_id}/columns', { params: { path: { run_id: props.runId } } }))
  .then((list) => (columns.value = list))
  .catch((caught) => (error.value = caught instanceof Error ? caught.message : String(caught)))
</script>

<style scoped>
.run-name {
  font-size: 22px;
}
</style>
