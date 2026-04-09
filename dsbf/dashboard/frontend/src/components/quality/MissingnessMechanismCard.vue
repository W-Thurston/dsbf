<!-- dsbf/dashboard/frontend/src/components/quality/MissingnessMechanismCard.vue

  Shows missingness mechanism analysis results (MCAR / MAR / MNAR) per column.
  Rendered after the Completeness dimension section in QualityTab.

  Design constraints:
  - Never show "MNAR" as a positive verdict - it is unverifiable from data
  - Always show "consistent with" language, never "is" or "confirmed"
  - Caveats are always visible, not hidden behind expansion
  - Confidence is always "moderate" or "low" - never "high"

  States:
  - not_run   : task absent from tasks prop
  - empty     : task ran, no columns with sufficient missingness (positive outcome)
  - ready     : one or more columns analysed
-->

<template>
  <div class="card mm-card">
    <div class="mm-header" @click="open = !open">
      <div class="mm-header-left">
        <span class="mm-title-wrap">
          <span class="mm-status-dot" :class="statusDotClass" />
          <span class="card-title">Missingness Mechanism Analysis</span>
          <TooltipIcon
            text="Classifies why values are missing in each column - whether missingness appears random (MCAR), related to other observed variables (MAR), or potentially related to the missing value itself (MNAR). Uses correlation patterns between null indicators and other columns. Results are probabilistic, not definitive."
            direction="down"
            align="left"
          />
        </span>
        <span v-if="state === 'ready'" class="mm-subtitle">
          {{ Object.keys(colResults).length }} column{{ Object.keys(colResults).length === 1 ? '' : 's' }} analysed
        </span>
      </div>
      <button class="mm-toggle-btn">{{ open ? '▲' : '▼' }}</button>
    </div>

    <Transition name="mm-expand">
      <div v-if="open" class="mm-body">

        <!-- Not run -->
        <div v-if="state === 'not_run'" class="es-not-run">
          Missingness mechanism analysis did not run for this profiling depth.
          Re-run at <strong>standard</strong> depth or higher to enable it.
        </div>

        <!-- Error -->
        <div v-else-if="state === 'error'" class="es-error">
          <span>⚠</span> {{ errorMessage }}
        </div>

        <!-- Empty - positive outcome -->
        <div v-else-if="state === 'empty'" class="es-empty">
          ✓ No columns with sufficient missingness to analyse (threshold: {{ minNullPct }}% null).
          <span class="mm-empty-note">
            Mechanism analysis is only meaningful when columns have missing values.
          </span>
        </div>

        <!-- Ready - show per-column assessments -->
        <template v-else>
          <!-- Global epistemic note -->
          <div class="mm-epistemic-banner">
            <span class="mm-epistemic-icon">ⓘ</span>
            <span>{{ epistemicNote }}</span>
          </div>

          <div
            v-for="(colData, col) in colResults"
            :key="col"
            class="mm-col-block"
          >
            <!-- Column header -->
            <div class="mm-col-header" @click="toggleCol(col)">
              <div class="mm-col-name">{{ col }}</div>
              <div class="mm-col-meta">
                <span class="mm-null-pct">{{ (colData.null_pct * 100).toFixed(1) }}% missing</span>
                <span
                  class="mm-assessment-badge"
                  :class="assessmentClass(colData.mechanism_assessment?.consistent_with)"
                >
                  {{ assessmentLabel(colData.mechanism_assessment?.consistent_with) }}
                </span>
                <span class="mm-confidence" :class="confidenceClass(colData.mechanism_assessment?.confidence)">
                  {{ colData.mechanism_assessment?.confidence ?? 'unknown' }} confidence
                </span>
              </div>
              <span class="mm-col-toggle">{{ openCols.has(col) ? '▲' : '▼' }}</span>
            </div>

            <!-- Column detail - expanded -->
            <Transition name="mm-col-expand">
              <div v-if="openCols.has(col)" class="mm-col-detail">

                <!-- Evidence summary -->
                <div v-if="colData.mechanism_assessment?.evidence_summary?.length" class="mm-section">
                  <div class="mm-section-label">Evidence</div>
                  <ul class="mm-evidence-list">
                    <li
                      v-for="(ev, i) in colData.mechanism_assessment.evidence_summary"
                      :key="i"
                      class="mm-evidence-item"
                    >{{ ev }}</li>
                  </ul>
                </div>

                <!-- Notable missingness correlations -->
                <div
                  v-if="notableCorrelations(colData).length"
                  class="mm-section"
                >
                  <div class="mm-section-label">
                    Missingness correlated with
                    <span class="mm-section-hint">(is_missing indicator vs observed column values)</span>
                  </div>
                  <div class="mm-corr-grid">
                    <div
                      v-for="corr in notableCorrelations(colData)"
                      :key="corr.col"
                      class="mm-corr-row"
                    >
                      <span class="mm-corr-col">{{ corr.col }}</span>
                      <span class="mm-corr-bar-wrap">
                        <span
                          class="mm-corr-bar"
                          :style="{ width: `${Math.abs(corr.r) * 100}%` }"
                          :class="corr.r < 0 ? 'mm-corr-bar--neg' : 'mm-corr-bar--pos'"
                        />
                      </span>
                      <span class="mm-corr-val" :class="corr.r < 0 ? 'val-neg' : 'val-pos'">
                        r={{ corr.r.toFixed(2) }}
                      </span>
                      <span class="mm-corr-p">p={{ corr.p.toFixed(3) }}</span>
                      <span class="mm-corr-strength">{{ corr.strength }}</span>
                    </div>
                  </div>
                </div>

                <!-- Caveats - always visible -->
                <div class="mm-section mm-caveats">
                  <div class="mm-section-label mm-caveats-label">⚠ Important caveats</div>
                  <ul class="mm-caveat-list">
                    <li
                      v-for="(cav, i) in colData.mechanism_assessment?.caveats ?? []"
                      :key="i"
                      class="mm-caveat-item"
                    >{{ cav }}</li>
                  </ul>
                </div>

              </div>
            </Transition>
          </div>
        </template>

      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  tasks: { type: Object, default: () => ({}) },
})

const open    = ref(false)
const openCols = ref(new Set())

function toggleCol(col) {
  const s = new Set(openCols.value)
  s.has(col) ? s.delete(col) : s.add(col)
  openCols.value = s
}

// ── State ─────────────────────────────────────────────────────────────────────

const taskResult  = computed(() => props.tasks?.missingness_mechanism_analysis ?? null)
const taskStatus  = computed(() => taskResult.value?.status ?? null)
const colResults  = computed(() => {
  const d = taskResult.value?.data ?? {}
  // Filter out non-column keys
  return Object.fromEntries(
    Object.entries(d).filter(([k]) => !k.startsWith('__'))
  )
})

const state = computed(() => {
  if (!taskResult.value)                        return 'not_run'
  if (taskStatus.value === 'error' ||
      taskStatus.value === 'failure')           return 'error'
  if (Object.keys(colResults.value).length === 0) return 'empty'
  return 'ready'
})

const errorMessage = computed(() =>
  taskResult.value?.error_metadata?.message ??
  taskResult.value?.summary?.message ??
  'This task encountered an error.'
)

const epistemicNote = computed(() =>
  taskResult.value?.summary?.epistemic_note ??
  'MCAR is the only mechanism with a statistical test. MAR is not directly ' +
  'testable. MNAR is unverifiable from observed data alone. All assessments ' +
  'are evidence-based hypotheses, not confirmations.'
)

const statusDotClass = computed(() => {
  if (state.value === 'empty' || state.value === 'not_run') return 'mm-dot--green'
  if (state.value === 'error') return 'mm-dot--amber'
  return 'mm-dot--amber'  // columns analysed = missingness present = worth noting
})

const minNullPct = computed(() => {
  const v = taskResult.value?.metadata?.min_null_pct ?? 0.01
  return (v * 100).toFixed(0)
})

// ── Helpers ───────────────────────────────────────────────────────────────────

function notableCorrelations(colData) {
  const corrs = colData.missingness_correlations ?? {}
  return Object.entries(corrs)
    .filter(([, v]) => Math.abs(v.correlation) >= 0.2)
    .sort((a, b) => Math.abs(b[1].correlation) - Math.abs(a[1].correlation))
    .slice(0, 8)
    .map(([col, v]) => ({
      col,
      r:        v.correlation,
      p:        v.p_value,
      strength: v.strength,
    }))
}

function assessmentLabel(consistentWith) {
  if (!consistentWith?.length) return 'Indeterminate'
  const labels = {
    mcar:          'Consistent with MCAR',
    mar:           'Consistent with MAR',
    indeterminate: 'Indeterminate',
  }
  return consistentWith.map(k => labels[k] ?? k.toUpperCase()).join(' / ')
}

function assessmentClass(consistentWith) {
  if (!consistentWith?.length || consistentWith.includes('indeterminate'))
    return 'mm-badge--gray'
  if (consistentWith.includes('mcar')) return 'mm-badge--green'
  if (consistentWith.includes('mar'))  return 'mm-badge--amber'
  return 'mm-badge--gray'
}

function confidenceClass(confidence) {
  if (confidence === 'moderate') return 'mm-conf--amber'
  if (confidence === 'low')      return 'mm-conf--gray'
  return 'mm-conf--gray'
}
</script>

<style scoped>
.mm-card { padding: 0; overflow: visible; }

/* ── Header ──────────────────────────────────────────────────────────────── */
.mm-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  cursor: pointer;
  user-select: none;
  transition: background 0.12s;
  gap: 12px;
}
.mm-header:hover { background: rgba(255,255,255,0.03); }

.mm-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.mm-title-wrap {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.mm-title-wrap .card-title { margin-bottom: 0; }
.mm-status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.mm-dot--green { background: #4ade80; box-shadow: 0 0 5px #4ade8055; }
.mm-dot--amber { background: #fbbf24; box-shadow: 0 0 5px #fbbf2455; }

.mm-subtitle { font-size: 12px; color: #64748b; }

.mm-toggle-btn {
  padding: 3px 8px;
  font-size: 11px;
  background: none;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #475569;
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.12s;
}
.mm-toggle-btn:hover { border-color: #60a5fa; color: #93c5fd; }

/* ── Body ────────────────────────────────────────────────────────────────── */
.mm-body {
  border-top: 1px solid #1e293b;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* ── Empty states ────────────────────────────────────────────────────────── */
.es-not-run { color: #475569; font-size: 13px; padding: 16px 0; text-align: center; }
.es-empty   { color: #4ade80; font-size: 13px; padding: 16px 0; text-align: center; display: flex; flex-direction: column; gap: 6px; align-items: center; }
.mm-empty-note { font-size: 11px; color: #475569; }
.es-error   { color: #f87171; font-size: 13px; padding: 12px; background: #3d0f0f; border-radius: 6px; border-left: 3px solid #f87171; display: flex; gap: 8px; }

/* ── Epistemic banner ────────────────────────────────────────────────────── */
.mm-epistemic-banner {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  background: #0f1e35;
  border: 1px solid #1e3a5f;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 12px;
  color: #93c5fd;
  line-height: 1.5;
}
.mm-epistemic-icon { flex-shrink: 0; font-size: 14px; margin-top: 1px; }

/* ── Column blocks ───────────────────────────────────────────────────────── */
.mm-col-block {
  border: 1px solid #1e293b;
  border-radius: 8px;
  overflow: hidden;
}

.mm-col-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  cursor: pointer;
  transition: background 0.12s;
  flex-wrap: wrap;
}
.mm-col-header:hover { background: rgba(255,255,255,0.03); }

.mm-col-name {
  font-family: ui-monospace, 'Cascadia Code', monospace;
  font-size: 13px;
  color: #93c5fd;
  font-weight: 600;
  min-width: 120px;
}

.mm-col-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  flex-wrap: wrap;
}

.mm-null-pct { font-size: 12px; color: #64748b; }

.mm-assessment-badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 9px;
  font-size: 11px;
  font-weight: 600;
  border: 1px solid;
}
.mm-badge--green { background: #0f2718; color: #4ade80; border-color: #4ade80; }
.mm-badge--amber { background: #3d2a00; color: #fbbf24; border-color: #fbbf24; }
.mm-badge--gray  { background: #1e293b; color: #94a3b8; border-color: #475569; }

.mm-confidence { font-size: 11px; }
.mm-conf--amber { color: #fbbf24; }
.mm-conf--gray  { color: #475569; }

.mm-col-toggle { font-size: 11px; color: #475569; margin-left: auto; }

/* ── Column detail ───────────────────────────────────────────────────────── */
.mm-col-detail {
  border-top: 1px solid #1e293b;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.mm-section { display: flex; flex-direction: column; gap: 6px; }

.mm-section-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #475569;
  display: flex;
  align-items: center;
  gap: 6px;
}
.mm-section-hint { font-size: 10px; color: #334155; text-transform: none; letter-spacing: 0; font-weight: 400; }

.mm-evidence-list, .mm-caveat-list {
  margin: 0;
  padding-left: 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.mm-evidence-item { font-size: 12px; color: #94a3b8; line-height: 1.4; }
.mm-caveat-item   { font-size: 12px; color: #64748b;  line-height: 1.4; }

/* ── Correlation bars ────────────────────────────────────────────────────── */
.mm-corr-grid { display: flex; flex-direction: column; gap: 4px; }

.mm-corr-row {
  display: grid;
  grid-template-columns: 140px 1fr 60px 70px 60px;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}

.mm-corr-col {
  font-family: ui-monospace, 'Cascadia Code', monospace;
  font-size: 11px;
  color: #94a3b8;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.mm-corr-bar-wrap {
  height: 6px;
  background: #1e293b;
  border-radius: 3px;
  overflow: hidden;
}
.mm-corr-bar {
  display: block;
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s;
}
.mm-corr-bar--pos { background: #60a5fa; }
.mm-corr-bar--neg { background: #f472b6; }

.mm-corr-val  { font-size: 11px; font-weight: 600; }
.val-pos { color: #60a5fa; }
.val-neg { color: #f472b6; }
.mm-corr-p        { font-size: 11px; color: #475569; }
.mm-corr-strength { font-size: 10px; color: #334155; text-transform: capitalize; }

/* ── Caveats ─────────────────────────────────────────────────────────────── */
.mm-caveats {
  background: #0f172a;
  border-radius: 6px;
  padding: 10px 12px;
}
.mm-caveats-label { color: #fbbf24; }

/* ── Transitions ─────────────────────────────────────────────────────────── */
.mm-expand-enter-active,
.mm-expand-leave-active,
.mm-col-expand-enter-active,
.mm-col-expand-leave-active {
  transition: opacity 0.18s, max-height 0.22s ease;
  max-height: 2000px;
  overflow: hidden;
}
.mm-expand-enter-from,
.mm-expand-leave-to,
.mm-col-expand-enter-from,
.mm-col-expand-leave-to {
  opacity: 0;
  max-height: 0;
}

/* ── Light theme ─────────────────────────────────────────────────────────── */
:global(.theme-light) .mm-header:hover   { background: rgba(0,0,0,0.02); }
:global(.theme-light) .mm-body           { border-top-color: #e2e8f0; }
:global(.theme-light) .mm-epistemic-banner { background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8; }
:global(.theme-light) .mm-col-block      { border-color: #e2e8f0; }
:global(.theme-light) .mm-col-header:hover { background: rgba(0,0,0,0.02); }
:global(.theme-light) .mm-col-detail     { border-top-color: #e2e8f0; }
:global(.theme-light) .mm-col-name       { color: #2563eb; }
:global(.theme-light) .mm-evidence-item  { color: #475569; }
:global(.theme-light) .mm-caveat-item    { color: #64748b; }
:global(.theme-light) .mm-caveats        { background: #fefce8; border: 1px solid #fde68a; }
:global(.theme-light) .mm-caveats-label  { color: #d97706; }
:global(.theme-light) .mm-corr-bar-wrap  { background: #e2e8f0; }
:global(.theme-light) .mm-corr-col       { color: #64748b; }
:global(.theme-light) .mm-corr-p         { color: #94a3b8; }
:global(.theme-light) .es-not-run        { color: #94a3b8; }
</style>
