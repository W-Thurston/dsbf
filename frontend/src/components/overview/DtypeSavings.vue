<template>
  <details
    v-if="rows.length"
    class="card"
    :open="dtypes.total_saving_bytes >= MB"
    data-section="dtypes"
  >
    <summary class="card-title">
      Dtype optimisations: about {{ formatBytes(dtypes.total_saving_bytes) }} to save
    </summary>
    <table class="data-table">
      <thead>
        <tr>
          <th scope="col">Column</th>
          <th scope="col">Now</th>
          <th scope="col">Suggested</th>
          <th scope="col" class="num">Saves</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="[name, s] in rows" :key="name">
          <th scope="row">{{ name }}</th>
          <td>{{ s.current_dtype }}</td>
          <td>{{ s.suggested_dtype }}</td>
          <td class="num">{{ formatBytes(s.estimated_saving_bytes) }}</td>
        </tr>
      </tbody>
    </table>
  </details>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { DtypeOptimizations } from '../../composables/useOverviewData'
import { formatBytes } from './overview'

const props = defineProps<{ dtypes: DtypeOptimizations }>()
const MB = 1024 * 1024
const rows = computed(() =>
  Object.entries(props.dtypes.suggestions).sort(
    (a, b) => b[1].estimated_saving_bytes - a[1].estimated_saving_bytes,
  ),
)
</script>
