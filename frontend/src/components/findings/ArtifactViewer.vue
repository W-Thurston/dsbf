<template>
  <div class="artifact">
    <button v-if="!result && !error" type="button" @click="load(0)">Load {{ artifactRef }}</button>
    <p v-if="error" class="error">{{ error }}</p>
    <template v-if="result && 'rows' in result">
      <table class="data-table">
        <thead>
          <tr>
            <th v-for="name in result.columns" :key="name" scope="col">{{ name }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, index) in result.rows" :key="result.offset + index">
            <td v-for="(cell, at) in row" :key="at">{{ cell }}</td>
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
    <pre v-else-if="result"><code>{{ JSON.stringify(result.value, null, 2) }}</code></pre>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { api, unwrap } from '../../api/client'
import type { components } from '../../generated/api'
import PagerNav from '../PagerNav.vue'

const props = defineProps<{ runId: string; artifactRef: string }>()

const ROWS = 20
type Artifact = components['schemas']['ArtifactRows'] | components['schemas']['ArtifactValue']

const result = ref<Artifact | null>(null)
const error = ref('')

async function load(offset: number) {
  error.value = ''
  try {
    result.value = await unwrap(
      api.GET('/v2/runs/{run_id}/artifacts/{ref}', {
        params: {
          path: { run_id: props.runId, ref: props.artifactRef },
          query: { limit: ROWS, offset },
        },
      }),
    )
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

// Another finding's artifact is a different one; nothing loads until it is asked for.
watch(
  () => props.artifactRef,
  () => {
    result.value = null
    error.value = ''
  },
)
</script>

<style scoped>
pre {
  max-height: 320px;
  overflow: auto;
}
</style>
