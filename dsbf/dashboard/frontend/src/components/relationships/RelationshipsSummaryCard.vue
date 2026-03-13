<template>
  <div v-if="hasAnyData" class="relationships-summary card">

    <!-- Associations block -->
    <div v-if="hasPairwise" class="summary-block">
      <div class="block-title">
        Pairwise Associations
        <TooltipIcon text="Association metrics computed across all eligible column pairs. Columns typed as id, datetime, or text are excluded." />
      </div>
      <div class="block-stats">
        <div class="stat-item">
          <span class="stat-value">{{ pairCount }}</span>
          <span class="stat-label">Total Pairs</span>
        </div>
        <div class="divider" />
        <div
          v-for="s in strengthBars"
          :key="s.key"
          class="stat-item"
        >
          <span class="stat-value" :class="`str-${s.key}`">{{ s.count }}</span>
          <span class="stat-label">{{ s.label }}</span>
        </div>
        <div class="divider" />
        <div class="strength-bar-wrap">
          <div class="strength-bar">
            <div
              v-for="s in strengthBars"
              :key="s.key"
              class="strength-segment"
              :class="`seg-${s.key}`"
              :style="{ width: s.pct + '%' }"
              :title="`${s.label}: ${s.count}`"
            />
          </div>
          <div class="bar-legend">
            <span v-for="s in strengthBars" :key="s.key" class="legend-item" :class="`str-${s.key}`">
              ● {{ s.label }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Warnings block: collinearity + leakage -->
    <div v-if="hasWarnings" class="summary-block warnings-block">
      <div class="block-title">Dataset-Level Warnings</div>
      <div class="warnings-row">

        <div v-if="collinearCount > 0" class="warning-chip warn-vif"
          :class="{ 'chip-active': activeFilter === 'collinearity' }"
          @click="$emit('filter-warnings', 'collinearity')">
          <span class="chip-icon">🔗</span>
          <span class="chip-count">{{ collinearCount }}</span>
          <span class="chip-label">Collinear Column{{ collinearCount !== 1 ? 's' : '' }}</span>
          <TooltipIcon :text="`${collinearCount} column(s) have VIF above threshold, indicating high multicollinearity. Click to filter the column browser.`" />
        </div>

        <div v-if="leakageCount > 0" class="warning-chip warn-leakage"
          :class="{ 'chip-active': activeFilter === 'leakage' }"
          @click="$emit('filter-warnings', 'leakage')">
          <span class="chip-icon">🚨</span>
          <span class="chip-count">{{ leakageCount }}</span>
          <span class="chip-label">Leakage Pair{{ leakageCount !== 1 ? 's' : '' }}</span>
          <TooltipIcon :text="`${leakageCount} column pair(s) flagged for potential data leakage. Click to filter the column browser.`" />
        </div>

        <div v-if="collinearCount === 0 && leakageCount === 0" class="no-warnings">
          ✅ No collinearity or leakage warnings detected.
        </div>

      </div>
    </div>

    <!-- Skipped columns note -->
    <div v-if="skippedColumns.length" class="skipped-note">
      <span class="skipped-icon">ℹ️</span>
      {{ skippedColumns.length }} column{{ skippedColumns.length !== 1 ? 's' : '' }} excluded from association analysis
      (id, datetime, text, or unknown type):
      <span class="skipped-cols">{{ skippedColumns.join(', ') }}</span>
    </div>

  </div>
</template>

<script setup>
import { computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  tasks:        { type: Object, default: () => ({}) },
  activeFilter: { type: String, default: null },
})

defineEmits(['filter-warnings'])

// ── Pairwise associations ─────────────────────────────────────────────────────

const pairwiseTask  = computed(() => props.tasks.compute_pairwise_associations ?? {})
const pairwiseSummary = computed(() => pairwiseTask.value.summary ?? {})
const hasPairwise   = computed(() => pairwiseTask.value.status === 'success' && pairCount.value > 0)
const pairCount     = computed(() => pairwiseSummary.value.pair_count ?? 0)

const rawStrengthCounts = computed(() => pairwiseSummary.value.strength_counts ?? {})

const strengthBars = computed(() => {
  const total = pairCount.value || 1
  return [
    { key: 'strong',     label: 'Strong',     count: rawStrengthCounts.value.strong     ?? 0 },
    { key: 'moderate',   label: 'Moderate',   count: rawStrengthCounts.value.moderate   ?? 0 },
    { key: 'weak',       label: 'Weak',       count: rawStrengthCounts.value.weak       ?? 0 },
    { key: 'negligible', label: 'Negligible', count: rawStrengthCounts.value.negligible ?? 0 },
  ].map(s => ({ ...s, pct: (s.count / total) * 100 }))
})

const skippedColumns = computed(
  () => pairwiseTask.value.metadata?.skipped_columns ?? []
)

// ── Collinearity ──────────────────────────────────────────────────────────────

const collinearColumns = computed(
  () => props.tasks.detect_collinear_features?.data?.collinear_columns ?? []
)
const collinearCount = computed(() => collinearColumns.value.length)

// ── Leakage ───────────────────────────────────────────────────────────────────

const leakagePairs = computed(
  () => props.tasks.detect_data_leakage?.data?.leakage_pairs ?? {}
)
const leakageCount = computed(() => Object.keys(leakagePairs.value).length)

// ── Visibility guards ─────────────────────────────────────────────────────────

const hasWarnings = computed(
  () =>
    props.tasks.detect_collinear_features?.status === 'success' ||
    props.tasks.detect_data_leakage?.status === 'success'
)

const hasAnyData = computed(() => hasPairwise.value || hasWarnings.value)
</script>

<style scoped>
.relationships-summary {
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* ── Block ───────────────────────────────────────────────────────────────── */
.summary-block {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.summary-block + .summary-block {
  padding-top: 16px;
  border-top: 1px solid #1e293b;
}

.block-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* ── Association stats row ───────────────────────────────────────────────── */
.block-stats {
  display: flex;
  align-items: center;
  gap: 24px;
  flex-wrap: wrap;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 52px;
}

.stat-value {
  font-size: 22px;
  font-weight: 700;
  color: #e2e8f0;
  line-height: 1;
}

.stat-label {
  font-size: 10px;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.divider {
  width: 1px;
  height: 36px;
  background: #1e293b;
  flex-shrink: 0;
}

/* Strength colours */
.str-strong     { color: #4ade80; }
.str-moderate   { color: #fb923c; }
.str-weak       { color: #60a5fa; }
.str-negligible { color: #94a3b8; }

/* ── Strength bar ─────────────────────────────────────────────────────────── */
.strength-bar-wrap {
  flex: 1;
  min-width: 160px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.strength-bar {
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  display: flex;
  background: #0f172a;
}

.strength-segment {
  height: 100%;
  transition: width 0.3s ease;
  min-width: 0;
}

.seg-strong     { background: #4ade80; }
.seg-moderate   { background: #fb923c; }
.seg-weak       { background: #60a5fa; }
.seg-negligible { background: #334155; }

.bar-legend {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.legend-item {
  font-size: 10px;
  letter-spacing: 0.3px;
}

/* ── Warnings row ────────────────────────────────────────────────────────── */
.warnings-block { }

.warnings-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.warning-chip {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 8px 14px;
  border-radius: 8px;
  cursor: pointer;
  border: 1px solid;
  transition: filter 0.15s;
  user-select: none;
}

.warning-chip:hover { filter: brightness(1.15); }

.chip-active {
  outline: 2px solid currentColor;
  outline-offset: 1px;
  filter: brightness(1.2);
}

.warn-vif {
  background: #2d1f0a;
  border-color: #fb923c;
  color: #fed7aa;
}

.warn-leakage {
  background: #2d1515;
  border-color: #f87171;
  color: #fca5a5;
}

.chip-icon  { font-size: 14px; }
.chip-count { font-size: 18px; font-weight: 700; }
.chip-label { font-size: 12px; font-weight: 500; }

.no-warnings {
  font-size: 13px;
  color: #4ade80;
}

/* ── Skipped note ─────────────────────────────────────────────────────────── */
.skipped-note {
  font-size: 11px;
  color: #475569;
  padding-top: 4px;
  border-top: 1px solid #1e293b;
  line-height: 1.5;
}

.skipped-icon { margin-right: 4px; }

.skipped-cols {
  font-family: monospace;
  color: #64748b;
}
</style>
