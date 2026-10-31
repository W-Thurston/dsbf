<template>
  <section class="card" aria-labelledby="transforms-title" data-role="transform-preview">
    <h2 id="transforms-title" class="card-title">Transformation preview</h2>
    <p v-if="state === 'loading'" class="loading">Loading…</p>
    <p v-else-if="!names.length" class="muted">
      {{ reason || 'No column is skewed enough to need a transform.' }}
    </p>
    <template v-else>
      <p class="muted">
        Skewness before and after each candidate; shorter is better. The recommended transform is
        starred.
      </p>
      <div class="columns">
        <div v-for="name in names" :key="name" class="column" :data-column="name">
          <h3>
            <router-link :to="{ name: 'column', params: { runId, column: name } }">{{
              name
            }}</router-link>
          </h3>
          <BarList :items="view(name).bars" :label="`Skewness of ${name}`" />
          <p v-if="!output?.columns[name]?.recommended" class="muted">
            No candidate brings it under the threshold.
          </p>
          <ul v-if="view(name).skipped.length" class="muted skipped">
            <li v-for="line in view(name).skipped" :key="line">{{ line }}</li>
          </ul>
        </div>
      </div>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../../api/client'
import BarList from '../overview/BarList.vue'
import { skewBars, type TransformationsOutput } from './readiness'

const props = defineProps<{ runId: string }>()

const output = ref<TransformationsOutput | null>(null)
const state = ref<'loading' | 'ready'>('loading')
const reason = ref('')

watch(
  () => props.runId,
  async (run) => {
    state.value = 'loading'
    output.value = null
    reason.value = ''
    try {
      const result = await unwrap(
        api.GET('/v2/runs/{run_id}/checks/{check_id}', {
          params: { path: { run_id: run, check_id: 'transformations' } },
        }),
      )
      if (result.status === 'ok') output.value = result.output as unknown as TransformationsOutput
      else
        reason.value = `The transformations check did not run${result.notes[0] ? `: ${result.notes[0].message}` : ''}.`
    } catch {
      output.value = null
    }
    state.value = 'ready'
  },
  { immediate: true },
)

const names = computed(() => Object.keys(output.value?.columns ?? {}))
const view = (name: string) => skewBars(output.value!.columns[name]!)
</script>

<style scoped>
.columns {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}
h3 {
  font-size: 14px;
  margin-bottom: 4px;
}
.skipped {
  font-size: 12px;
  padding-left: 18px;
}
</style>
