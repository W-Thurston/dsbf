<template>
  <section class="card" aria-labelledby="sample-title" data-section="sample">
    <h2 id="sample-title" class="card-title">Dataset sample</h2>
    <p v-if="state === 'loading'" class="loading">Loading…</p>
    <p v-else-if="state === 'none'" class="muted" data-role="no-snapshot">
      This run kept no data snapshot, so there are no rows to show.
    </p>
    <p v-else-if="error" class="error">{{ error }}</p>
    <template v-else-if="rows">
      <label class="size"
        >Rows
        <select v-model.number="size" aria-label="Rows shown">
          <option v-for="n in SIZES" :key="n" :value="n">{{ n }}</option>
        </select>
      </label>
      <div class="scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th v-for="name in rows.columns" :key="name" scope="col">{{ name }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in rows.rows.slice(0, size)" :key="i">
              <td v-for="(cell, j) in row" :key="j" :class="{ null: cell === null }">
                {{ cell === null ? 'null' : String(cell) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </section>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { api } from '../../api/client'
import type { components } from '../../generated/api'

const props = defineProps<{ runId: string }>()

const SIZES = [5, 10, 25, 50]
const size = ref(10)
const rows = ref<components['schemas']['ArtifactRows'] | null>(null)
const state = ref<'loading' | 'none' | 'ready'>('loading')
const error = ref('')

watch(
  () => props.runId,
  async (run) => {
    state.value = 'loading'
    error.value = ''
    rows.value = null
    try {
      const { data, response } = await api.GET('/v2/runs/{run_id}/snapshot', {
        params: { path: { run_id: run }, query: { limit: Math.max(...SIZES) } },
      })
      if (response.status === 404) state.value = 'none'
      else if (data) {
        rows.value = data
        state.value = 'ready'
      } else {
        error.value = `the sample could not be read (${response.status})`
        state.value = 'ready'
      }
    } catch (caught) {
      error.value = caught instanceof Error ? caught.message : String(caught)
      state.value = 'ready'
    }
  },
  { immediate: true },
)
</script>

<style scoped>
.size {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  margin-bottom: 8px;
}
.scroll {
  overflow-x: auto;
}
.null {
  color: var(--text-muted);
  font-style: italic;
}
</style>
