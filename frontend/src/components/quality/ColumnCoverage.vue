<template>
  <section class="card" aria-labelledby="coverage-title">
    <h2 id="coverage-title" class="card-title">Coverage by column</h2>
    <p class="muted">
      How many of the {{ checksCompleted }} checks that completed analysed each column. A check
      leaves out a column it cannot read, such as a text column for a numeric test.
    </p>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="!coverage" class="loading">Loading…</p>
    <table v-else class="data-table">
      <thead>
        <tr>
          <th scope="col">Column</th>
          <th scope="col">Checks that analysed it</th>
          <th scope="col">Share</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in coverage.columns" :key="row.column">
          <td>
            <router-link :to="{ name: 'column', params: { runId, column: row.column } }">{{
              row.column
            }}</router-link>
          </td>
          <td>{{ row.examined_by }} of {{ row.checks_completed }}</td>
          <td>{{ percent(row.checks_completed ? row.examined_by / row.checks_completed : 0) }}</td>
        </tr>
      </tbody>
    </table>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../../api/client'
import type { components } from '../../generated/api'
import { percent } from './quality'

const props = defineProps<{ runId: string }>()
const coverage = ref<components['schemas']['RunCoverage'] | null>(null)
const error = ref('')
const checksCompleted = computed(() => coverage.value?.columns[0]?.checks_completed ?? 0)

watch(
  () => props.runId,
  async () => {
    coverage.value = null
    error.value = ''
    try {
      coverage.value = await unwrap(
        api.GET('/v2/runs/{run_id}/coverage', { params: { path: { run_id: props.runId } } }),
      )
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    }
  },
  { immediate: true },
)
</script>
