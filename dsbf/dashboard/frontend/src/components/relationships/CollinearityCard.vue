<template>
  <div v-if="items.length" class="collinearity-card card">
    <div class="card-title">⚠️ Relationship Warnings</div>
    <div class="warning-list">
      <div v-for="item in items" :key="item.code" class="warning-item" :class="`warn-${item.level}`">
        <div class="warn-header">
          <span class="warn-icon">{{ item.icon }}</span>
          <span class="warn-title">{{ item.title }}</span>
          <span class="warn-level">{{ item.level }}</span>
        </div>
        <p class="warn-body">{{ item.body }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  colA:   { type: String, required: true },
  colB:   { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
})

const items = computed(() => {
  const out = []

  // ── Collinearity (VIF) ─────────────────────────────────────────────────────
  const vif = props.tasks.detect_collinear_features?.data?.vif_scores ?? {}
  const collinearCols = props.tasks.detect_collinear_features?.data?.collinear_columns ?? []
  const vifThreshold  = props.tasks.detect_collinear_features?.metadata?.vif_threshold ?? 10

  for (const col of [props.colA, props.colB]) {
    if (col in vif) {
      const score = vif[col]
      if (score > vifThreshold) {
        out.push({
          code:  `vif_${col}`,
          level: score > 30 ? 'critical' : 'warning',
          icon:  '🔗',
          title: `High Multicollinearity — ${col}`,
          body:  `VIF = ${score.toFixed(1)}. Values above ${vifThreshold} indicate this column is highly correlated with other features, which can cause instability in linear models. Consider dropping one of the collinear columns or using regularization.`,
        })
      }
    }
  }

  // ── Leakage ────────────────────────────────────────────────────────────────
  const leakagePairs = props.tasks.detect_data_leakage?.data?.leakage_pairs ?? {}
  const pairKey1 = `${props.colA}|${props.colB}`
  const pairKey2 = `${props.colB}|${props.colA}`

  if (pairKey1 in leakagePairs || pairKey2 in leakagePairs) {
    const info = leakagePairs[pairKey1] ?? leakagePairs[pairKey2]
    out.push({
      code:  'leakage',
      level: 'critical',
      icon:  '🚨',
      title: 'Potential Data Leakage',
      body:  `This pair was flagged as a potential leakage risk${info ? `: ${info}` : '. These columns may be derived from each other or share a causal relationship that would not exist at prediction time.'}`,
    })
  }

  return out
})
</script>

<style scoped>
.collinearity-card { padding: 16px 20px; }
.card-title { font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.4px; }
.warning-list { display: flex; flex-direction: column; gap: 10px; }
.warning-item { border-radius: 8px; padding: 12px 14px; border-left: 3px solid; }
.warn-critical { background: #2d1515; border-color: #f87171; }
.warn-warning  { background: #2d1f0a; border-color: #fb923c; }
.warn-header   { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.warn-icon     { font-size: 15px; }
.warn-title    { font-size: 13px; font-weight: 600; color: #e2e8f0; flex: 1; }
.warn-level    { font-size: 10px; text-transform: uppercase; letter-spacing: 0.4px; color: #94a3b8; }
.warn-body     { font-size: 12px; color: #94a3b8; line-height: 1.5; margin: 0; }
</style>
