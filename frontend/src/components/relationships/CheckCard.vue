<template>
  <section class="card" :aria-labelledby="`${checkId}-title`">
    <h2 :id="`${checkId}-title`" class="card-title">{{ title }}</h2>
    <p v-if="state === 'loading'" class="loading">Loading…</p>
    <p v-else-if="state === 'missing'" class="muted">{{ title }} did not run in this run.</p>
    <template v-else-if="result">
      <p v-if="result.status !== 'ok'" class="muted">
        Status {{ result.status
        }}<template v-if="result.notes.length">: {{ result.notes[0]?.message }}</template>
      </p>
      <slot v-else :output="result.output ?? {}" />
    </template>
  </section>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { api, unwrap } from '../../api/client'
import type { components } from '../../generated/api'

const props = defineProps<{ runId: string; checkId: string; title: string }>()

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
