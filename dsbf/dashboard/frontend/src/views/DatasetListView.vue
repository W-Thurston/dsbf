<template>
  <div>
    <h1 class="page-title">Datasets</h1>

    <div v-if="loading" class="loading">Loading datasets…</div>
    <div v-else-if="error" class="error">{{ error }}</div>

    <div v-else class="dataset-grid">
      <router-link
        v-for="ds in datasets"
        :key="ds.id"
        :to="`/datasets/${ds.name}`"
        class="dataset-card card"
      >
        <div class="ds-name">{{ ds.name }}</div>
        <div class="ds-meta">
          <span class="meta-item">
            <span class="meta-label">Runs</span>
            <span class="meta-value">{{ ds.run_count }}</span>
          </span>
          <span class="meta-item">
            <span class="meta-label">Latest</span>
            <span class="meta-value">{{ formatDate(ds.latest_ran_at) }}</span>
          </span>
          <span class="meta-item">
            <span class="meta-label">Quality</span>
            <span class="meta-value" :class="qualityClass(ds.latest_quality_score)">
              {{ ds.latest_quality_score ?? '-' }}
            </span>
          </span>
        </div>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { listDatasets } from '../api.js'
import { formatDate, qualityClass } from '../utils.js'

const datasets = ref([])
const loading  = ref(true)
const error    = ref(null)

onMounted(async () => {
  try {
    datasets.value = await listDatasets()
  } catch (e) {
    error.value = `Failed to load datasets: ${e.message}`
  } finally {
    loading.value = false
  }
})


</script>

<style scoped>
.page-title {
  font-size: 22px;
  font-weight: 600;
  margin-bottom: 20px;
  color: #f1f5f9;
}

.dataset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}

.dataset-card {
  display: block;
  color: inherit;
  transition: border-color 0.15s, transform 0.15s;
}

.dataset-card:hover {
  border-color: #60a5fa;
  transform: translateY(-2px);
  text-decoration: none;
}

.ds-name {
  font-size: 17px;
  font-weight: 600;
  color: #f1f5f9;
  margin-bottom: 14px;
}

.ds-meta {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.meta-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.meta-label {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.meta-value {
  font-size: 15px;
  font-weight: 600;
  color: #e2e8f0;
}

.score-excellent { color: #4ade80; }
.score-good      { color: #34d399; }
.score-warn      { color: #fb923c; }
.score-poor      { color: #f87171; }
</style>
