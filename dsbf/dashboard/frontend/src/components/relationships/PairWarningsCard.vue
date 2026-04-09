<!-- dsbf/dashboard/frontend/src/components/relationships/PairWarningsCard.vue

  Shows collinearity (VIF) and leakage warnings for a selected column pair.
  Language is observational — describes what the data shows, not what to do about it.
  Preparation recommendations belong in the ML Readiness tab.

  Always renders — shows an empty state when no warnings exist so the
  panel structure is consistent regardless of pair selection.

  Props
  ─────
  colA  : String
  colB  : String
  tasks : Object
-->
<template>
  <div class="pair-warnings-card card">

    <div class="card-title">
      Relationship Warnings
      <TooltipIcon
        text="Flags raised for this specific column pair from collinearity analysis (VIF) and near-perfect correlation detection. These describe structural properties of the data, not recommendations."
        direction="down"
        align="left"
      />
    </div>

    <!-- Clean state -->
    <div v-if="!items.length" class="warnings-clean">
      ✓ No relationship warnings for this pair.
    </div>

    <!-- Warnings list -->
    <div v-else class="warning-list">
      <div
        v-for="item in items"
        :key="item.code"
        class="warning-item"
        :class="`warn-${item.level}`"
      >
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
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  colA:  { type: String, required: true },
  colB:  { type: String, required: true },
  tasks: { type: Object, default: () => ({}) },
})

const items = computed(() => {
  const out = []

  // ── Collinearity (VIF) ────────────────────────────────────────────────────
  const vif          = props.tasks.detect_collinear_features?.data?.vif_scores ?? {}
  const vifThreshold = props.tasks.detect_collinear_features?.metadata?.vif_threshold ?? 10

  for (const col of [props.colA, props.colB]) {
    if (col in vif) {
      const score = vif[col]
      if (score > vifThreshold) {
        out.push({
          code:  `vif_${col}`,
          level: score > 30 ? 'critical' : 'warning',
          icon:  '🔗',
          title: `High Collinearity — ${col}`,
          body:  `VIF = ${score.toFixed(1)}. This column's variance is largely explained by other columns in the dataset — it does not carry fully independent information. The threshold is ${vifThreshold}.`,
        })
      }
    }
  }

  // ── Leakage ───────────────────────────────────────────────────────────────
  const leakagePairs = props.tasks.detect_data_leakage?.data?.leakage_pairs ?? {}
  const threshold    = props.tasks.detect_data_leakage?.metadata?.correlation_threshold ?? 0.99
  const pairKey1     = `${props.colA}|${props.colB}`
  const pairKey2     = `${props.colB}|${props.colA}`

  if (pairKey1 in leakagePairs || pairKey2 in leakagePairs) {
    const corr = leakagePairs[pairKey1] ?? leakagePairs[pairKey2]
    const rStr = typeof corr === 'number' ? ` (r = ${Math.abs(corr).toFixed(4)})` : ''
    out.push({
      code:  'leakage',
      level: 'critical',
      icon:  '⚠',
      title: 'Near-Perfect Correlation',
      body:  `These two columns are correlated at r ≥ ${threshold}${rStr}. They appear to encode the same or nearly the same information — one may be derived from the other, or both may share a common source.`,
    })
  }

  return out
})
</script>

<style scoped>
.pair-warnings-card {
  padding: 16px 20px;
  overflow: visible;
}

.card-title {
  font-size: 13px;
  font-weight: 600;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.warnings-clean {
  font-size: 13px;
  color: #4ade80;
  padding: 4px 0;
}

.warning-list { display: flex; flex-direction: column; gap: 10px; }

.warning-item {
  border-radius: 8px;
  padding: 12px 14px;
  border-left: 3px solid;
}
.warn-critical { background: #2d1515; border-color: #f87171; }
.warn-warning  { background: #2d1f0a; border-color: #fbbf24; }

.warn-header   { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.warn-icon     { font-size: 15px; }
.warn-title    { font-size: 13px; font-weight: 600; color: #e2e8f0; flex: 1; }
.warn-level    { font-size: 10px; text-transform: uppercase; letter-spacing: 0.4px; color: #94a3b8; }
.warn-body     { font-size: 12px; color: #94a3b8; line-height: 1.55; margin: 0; }
</style>
