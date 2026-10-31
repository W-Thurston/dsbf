<template>
  <section class="card" :aria-labelledby="`${checkId}-title`">
    <h2 :id="`${checkId}-title`" class="card-title">{{ title }}</h2>
    <p v-if="state === 'loading'" class="loading">Loading…</p>
    <p v-else-if="state === 'missing'" class="muted" data-role="check-missing">
      {{ missingNote ?? `${title} did not run in this run.` }}
    </p>
    <template v-else-if="result">
      <p v-if="result.status !== 'ok'" class="muted">
        {{ statusWords(result.status) }}
        <template v-if="result.notes.length">{{ result.notes[0]?.message }}</template>
      </p>
      <slot v-else :output="result.output ?? {}" />
    </template>
  </section>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { api, unwrap } from '../../api/client'
import type { components } from '../../generated/api'

const props = defineProps<{
  runId: string
  checkId: string
  title: string
  /** Why the check is absent from a run, when the default "did not run" line is too little. */
  missingNote?: string
}>()

const STATUS_WORDS: Record<string, string> = {
  not_applicable: 'Not calculated: this check does not apply to this dataset.',
  unavailable: 'Not calculated: a requirement of this check is not available.',
  skipped: 'Not calculated: the check was skipped.',
  error: 'Not calculated: the check raised an error.',
  timeout: 'Not calculated: the check ran out of time.',
}
const statusWords = (status: string) => STATUS_WORDS[status] ?? `Status ${status}.`

const result = ref<components['schemas']['CheckResult'] | null>(null)
const state = ref<'loading' | 'missing' | 'ready'>('loading')

watch(
  () => [props.runId, props.checkId],
  async () => {
    state.value = 'loading'
    try {
      result.value = await unwrap(
        api.GET('/v2/runs/{run_id}/checks/{check_id}', {
          params: { path: { run_id: props.runId, check_id: props.checkId } },
        }),
      )
      state.value = 'ready'
    } catch {
      result.value = null
      state.value = 'missing'
    }
  },
  { immediate: true },
)
</script>
