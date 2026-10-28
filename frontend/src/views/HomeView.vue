<template>
  <section class="card">
    <h1 class="card-title">Datasets</h1>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="datasets === null" class="loading">Loading…</p>
    <ul v-else>
      <li v-for="dataset in datasets" :key="dataset.key">{{ dataset.name }}</li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, unwrap } from '../api/client'
import type { components } from '../generated/api'

type Dataset = components['schemas']['DatasetSummary']

const datasets = ref<Dataset[] | null>(null)
const error = ref('')

onMounted(async () => {
  try {
    datasets.value = (await unwrap(api.GET('/v2/datasets'))).items
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
})
</script>
