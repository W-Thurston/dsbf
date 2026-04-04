<template>
  <div>
    <div class="page-header">
      <h1 class="page-title">Datasets</h1>
      <div class="search-wrap">
        <span class="search-icon">⌕</span>
        <input
          v-model="query"
          class="search-input"
          type="text"
          placeholder="Search datasets…"
          autocomplete="off"
          spellcheck="false"
        />
        <button v-if="query" class="search-clear" @click="query = ''" title="Clear">✕</button>
      </div>
    </div>

    <div v-if="loading" class="loading">Loading datasets…</div>
    <div v-else-if="error" class="error">{{ error }}</div>

    <template v-else>
      <div v-if="filtered.length === 0" class="empty-state">
        No datasets match <strong>{{ query }}</strong>
      </div>

      <div v-else class="dataset-grid">
        <router-link
          v-for="ds in filtered"
          :key="ds.id"
          :to="`/datasets/${ds.name}`"
          class="dataset-card card"
        >
          <div class="ds-name" v-html="highlight(ds.name, query)" />
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

      <div class="result-count" v-if="query">
        {{ filtered.length }} of {{ datasets.length }} datasets
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { listDatasets } from '../api.js'

const datasets = ref([])
const loading  = ref(true)
const error    = ref(null)
const query    = ref('')

onMounted(async () => {
  try {
    datasets.value = await listDatasets()
  } catch (e) {
    error.value = `Failed to load datasets: ${e.message}`
  } finally {
    loading.value = false
  }
})

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return datasets.value
  return datasets.value.filter(ds => ds.name.toLowerCase().includes(q))
})

function highlight(text, q) {
  if (!q.trim()) return text
  const escaped = q.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return text.replace(new RegExp(`(${escaped})`, 'gi'), '<mark>$1</mark>')
}

function formatDate(iso) {
  if (!iso) return '-'
  return new Date(iso).toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
  })
}

function qualityClass(score) {
  if (score == null) return ''
  if (score >= 90) return 'score-excellent'
  if (score >= 70) return 'score-good'
  if (score >= 40) return 'score-warn'
  return 'score-poor'
}
</script>

<style scoped>
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  gap: 16px;
  flex-wrap: wrap;
}

.page-title {
  font-size: 22px;
  font-weight: 600;
  color: #f1f5f9;
  margin: 0;
}

.search-wrap {
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: 10px;
  font-size: 16px;
  color: #64748b;
  pointer-events: none;
  line-height: 1;
}

.search-input {
  width: 240px;
  padding: 7px 32px 7px 30px;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  color: #e2e8f0;
  font-size: 14px;
  outline: none;
  transition: border-color 0.15s;
}

.search-input:focus {
  border-color: #60a5fa;
}

.search-input::placeholder { color: #64748b; }

.search-clear {
  position: absolute;
  right: 8px;
  background: none;
  border: none;
  color: #64748b;
  cursor: pointer;
  font-size: 12px;
  padding: 2px 4px;
  line-height: 1;
}
.search-clear:hover { color: #e2e8f0; }

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

.empty-state {
  color: #64748b;
  font-size: 14px;
  padding: 40px 0;
  text-align: center;
}

.empty-state strong { color: #94a3b8; }

.result-count {
  margin-top: 16px;
  font-size: 12px;
  color: #64748b;
  text-align: right;
}

.score-excellent { color: #4ade80; }
.score-good      { color: #34d399; }
.score-warn      { color: #fb923c; }
.score-poor      { color: #f87171; }

:deep(mark) {
  background: #facc15;
  color: #0f172a;
  border-radius: 2px;
  padding: 0 1px;
}
</style>
