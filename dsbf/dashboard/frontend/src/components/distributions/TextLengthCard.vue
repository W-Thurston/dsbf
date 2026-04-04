<template>
  <div v-if="hasData" class="card text-length-card">
    <div class="card-title">String Length Summary</div>
    <div class="tl-grid">
      <div class="tl-item" v-for="item in items" :key="item.label">
        <span class="tl-label" :title="item.tooltip">{{ item.label }}</span>
        <span class="tl-value">{{ item.value }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  column: { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
})

const textStats = computed(() =>
  props.tasks.summarize_text_fields?.data?.[props.column] ?? null
)
const catStats = computed(() =>
  props.tasks.categorical_length_stats?.data?.[props.column] ?? null
)

const hasData = computed(() => textStats.value != null || catStats.value != null)

function fmt(v, dp = 1) {
  if (v == null) return '-'
  return typeof v === 'number' ? v.toFixed(dp) : String(v)
}

const items = computed(() => {
  const ts = textStats.value
  const cs = catStats.value
  const out = []

  if (cs) {
    out.push(
      { label: 'Min Length',  tooltip: 'Shortest string value (character count).', value: fmt(cs.min_length, 0) },
      { label: 'Max Length',  tooltip: 'Longest string value (character count).', value: fmt(cs.max_length, 0) },
      { label: 'Avg Length',  tooltip: 'Mean string length across all non-null rows.', value: fmt(cs.mean_length) },
    )
  }

  if (ts) {
    out.push(
      { label: 'Avg Words',   tooltip: 'Average number of whitespace-delimited words per value.', value: fmt(ts.avg_word_count) },
      { label: 'Avg Word Len',tooltip: 'Average length of individual words.', value: fmt(ts.avg_word_length) },
      { label: 'Has Symbols', tooltip: 'Whether any values contain non-alphanumeric characters.', value: ts.contains_symbols ? 'Yes' : 'No' },
    )
  }

  return out
})
</script>

<style scoped>
.text-length-card { margin-top: 0; }

.tl-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
}

.tl-item {
  flex: 1 1 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 14px;
  border-right: 1px solid #1e293b;
  gap: 4px;
}
.tl-item:last-child { border-right: none; }

.tl-label {
  font-size: 10px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  white-space: nowrap;
  cursor: default;
}

.tl-value {
  font-size: 14px;
  font-weight: 600;
  color: #e2e8f0;
  font-family: monospace;
}
</style>
