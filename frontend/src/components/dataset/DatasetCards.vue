<template>
  <div class="cards" data-section="dataset-cards">
    <router-link
      v-for="dataset in datasets"
      :key="dataset.key"
      :to="{ name: 'dataset', params: { datasetKey: dataset.key } }"
      class="card dataset-card"
      :data-dataset="dataset.key"
    >
      <h2 class="name">{{ dataset.name }}</h2>
      <DimensionDots :levels="dataset.dimension_levels" />
      <dl>
        <div>
          <dt>Runs</dt>
          <dd>{{ dataset.n_runs }}</dd>
        </div>
        <div>
          <dt>Last run</dt>
          <dd>{{ formatDateTime(dataset.last_run_at) }}</dd>
        </div>
        <div>
          <dt>Highest</dt>
          <dd><SeverityBadge :severity="dataset.last_max_severity" /></dd>
        </div>
      </dl>
    </router-link>
  </div>
</template>

<script setup lang="ts">
import { formatDateTime } from '../../format'
import type { components } from '../../generated/api'
import SeverityBadge from '../SeverityBadge.vue'
import DimensionDots from './DimensionDots.vue'

defineProps<{ datasets: components['schemas']['DatasetSummary'][] }>()
</script>

<style scoped>
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}
.dataset-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  color: inherit;
  text-decoration: none;
}
.dataset-card:hover,
.dataset-card:focus-visible {
  border-color: var(--text-muted);
}
.name {
  margin: 0;
  font-size: 16px;
  overflow-wrap: anywhere;
}
dl {
  display: grid;
  gap: 4px;
  margin: 0;
}
dl div {
  display: flex;
  justify-content: space-between;
  gap: 8px;
}
dt {
  color: var(--text-muted);
}
dd {
  margin: 0;
}
</style>
