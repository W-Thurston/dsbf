<!-- dsbf/dashboard/frontend/src/components/distributions/TransformationPreviewCard.vue

  Shows transformation options for skewed continuous columns.
  Layout matches the action-row format: what / trade-off / after this / link,
  plus the unique computed outcome (before → after skewness) where available.

  Used in:
    - DistributionsTab (EDA context, full card)
    - MlReadinessTab   (ML prep context, embedded, no before-strip)

  Props
  ─────
  column : String
  tasks  : Object
  embedded : Boolean — when true, strips the card shell (via CSS :deep in parent)
-->

<template>
  <div class="card tx-card">
    <div class="card-title">
      Transformation Preview
      <TooltipIcon
        text="For skewed columns, shows how common transformations would affect the distribution's skewness. The before → after comparison uses the actual computed statistics. Purely informational — no data is modified."
        align="left"
        direction="down"
      />
    </div>

    <div v-if="state === 'not_run'" class="tx-not-run">
      Transformation preview did not run for this column. This analysis only runs
      for columns that exceed the skewness threshold.
    </div>

    <template v-else>
      <!-- Transform options -->
      <div class="tx-list">
        <div
          v-for="tx in transforms"
          :key="tx.key"
          class="tx-item"
          :class="{
            'tx-item--recommended': tx.recommended,
            'tx-item--skipped':     tx.skipped,
          }"
        >
          <!-- Header row: name + badge -->
          <div class="tx-item-header">
            <span class="tx-name">{{ tx.display }}</span>
            <span v-if="tx.recommended"  class="tx-badge-recommended">recommended</span>
            <span v-else-if="tx.skipped" class="tx-badge-skipped">not applicable</span>
          </div>

          <!-- Skipped: enriched explanation -->
          <p v-if="tx.skipped" class="tx-skip-body">{{ enrichSkipReason(tx.skip_reason) }}</p>

          <!-- Has computed results: rich metadata + outcome -->
          <template v-else-if="tx.after_stats">
            <p v-if="meta(tx.key)?.what" class="tx-what">{{ meta(tx.key).what }}</p>

            <div v-if="meta(tx.key)?.tradeoff" class="tx-meta-row">
              <span class="tx-meta-label">Trade-off</span>
              <span class="tx-meta-text">{{ meta(tx.key).tradeoff }}</span>
            </div>

            <div v-if="meta(tx.key)?.after" class="tx-meta-row">
              <span class="tx-meta-label">After this</span>
              <span class="tx-meta-text">{{ meta(tx.key).after }}</span>
            </div>

            <a
              v-if="meta(tx.key)?.ref"
              :href="meta(tx.key).ref.url"
              target="_blank"
              rel="noopener noreferrer"
              class="tx-ref"
            >↗ {{ meta(tx.key).ref.label }}</a>

            <!-- Computed outcome: before → after for key stats -->
            <div class="tx-outcome">
              <div class="tx-outcome-header">
                <span class="tx-outcome-col-label"></span>
                <span class="tx-outcome-col-label">Before</span>
                <span class="tx-outcome-col-label">After</span>
                <span class="tx-outcome-col-label"></span>
              </div>
              <div
                v-for="stat in outcomeStats(tx)"
                :key="stat.key"
                class="tx-outcome-row"
              >
                <span class="tx-outcome-stat-label">{{ stat.label }}</span>
                <span class="tx-outcome-stat-before">{{ stat.before }}</span>
                <span
                  class="tx-outcome-stat-after"
                  :class="stat.cls"
                >{{ stat.after }}</span>
                <span
                  v-if="stat.pct != null"
                  class="tx-reduction"
                  :class="stat.pct > 0 ? 'tx-reduction--good' : 'tx-reduction--bad'"
                >{{ stat.pct > 0 ? '↓' : '↑' }}{{ Math.abs(stat.pct).toFixed(0) }}%</span>
                <span v-else class="tx-reduction" />
              </div>
            </div>
          </template>

          <!-- Has metadata but no computed results -->
          <template v-else-if="meta(tx.key)">
            <p class="tx-what">{{ meta(tx.key).what }}</p>
          </template>
        </div>
      </div>

      <div class="tx-footer">Purely informational — no data has been modified by DSBF.</div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'
import { TRANSFORM_META, enrichSkipReason } from '../../utils/actionMeta.js'

const props = defineProps({
  column:   { type: String, required: true },
  tasks:    { type: Object, default: () => ({}) },
  embedded: { type: Boolean, default: false },
})

function fmtN(v) {
  if (v == null) return '—'
  const n = Number(v)
  if (!isFinite(n)) return '—'
  if (Math.abs(n) >= 10000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) >= 10)    return n.toFixed(2)
  return n.toPrecision(4).replace(/\.?0+$/, '')
}

function meta(key) {
  return TRANSFORM_META[key] ?? null
}

function isImproved(tx) {
  return tx.after_stats?.skewness != null &&
    Math.abs(tx.after_stats.skewness) < Math.abs(colData.value?.original_skewness ?? 0)
}

// Build the before/after stat rows shown in the outcome grid
const OUTCOME_STATS = [
  { key: 'skewness', label: 'Skewness', isPrimary: true },
  { key: 'mean',     label: 'Mean',     isPrimary: false },
  { key: 'std',      label: 'Std Dev',  isPrimary: false },
  { key: 'p5',       label: 'p5',       isPrimary: false },
  { key: 'p95',      label: 'p95',      isPrimary: false },
]

function outcomeStats(tx) {
  if (!tx.after_stats || !colData.value) return []
  const before = colData.value.before_stats ?? {}
  return OUTCOME_STATS.map(s => {
    const bVal = s.key === 'skewness' ? colData.value.original_skewness : before[s.key]
    const aVal = tx.after_stats[s.key]
    const improved = s.isPrimary && aVal != null && Math.abs(aVal) < Math.abs(bVal ?? 0)
    const worse    = s.isPrimary && aVal != null && Math.abs(aVal) >= Math.abs(bVal ?? 0)
    const pct = s.isPrimary && tx.skew_reduction_pct != null ? tx.skew_reduction_pct : null
    return {
      key:    s.key,
      label:  s.label,
      before: fmtN(bVal),
      after:  fmtN(aVal),
      cls:    improved ? 'tx-improved' : worse ? 'tx-worse' : '',
      pct,
    }
  }).filter(s => s.before !== '—' || s.after !== '—')
}

const colData = computed(() =>
  props.tasks?.transformation_preview?.data?.[props.column] ?? null
)

const state = computed(() => colData.value ? 'ready' : 'not_run')

const transforms = computed(() => {
  if (!colData.value?.transforms) return []
  const origSkew = Math.abs(colData.value?.original_skewness ?? 0)
  return Object.entries(colData.value.transforms).map(([key, tx]) => {
    const afterSkew = tx.after_stats?.skewness != null ? Math.abs(tx.after_stats.skewness) : null
    const actuallyBetter = afterSkew != null && afterSkew < origSkew
    return {
      key,
      display:            tx.display      ?? key,
      suitable_for:       tx.suitable_for ?? '',
      skipped:            tx.skipped      ?? false,
      skip_reason:        tx.skip_reason  ?? '',
      after_stats:        tx.after_stats  ?? null,
      skew_reduction_pct: tx.skew_reduction_pct ?? null,
      recommended:        (tx.recommended ?? false) && actuallyBetter,
    }
  })
})
</script>

<style scoped>
.tx-card { display: flex; flex-direction: column; gap: 14px; }
.tx-not-run { font-size: 13px; color: #64748b; line-height: 1.6; }

/* ── Transform list ────────────────────────────────────────────────────────── */
.tx-list { display: flex; flex-direction: column; gap: 8px; }

.tx-item {
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: border-color 0.15s;
}
.tx-item--recommended { border-color: #4ade80; background: rgba(74,222,128,0.03); }
.tx-item--skipped     { opacity: 0.55; }

.tx-item-header { display: flex; align-items: center; gap: 8px; }

.tx-name {
  font-size: 13px; font-weight: 600; color: #e2e8f0;
  font-family: ui-monospace, monospace;
}
.tx-badge-recommended {
  font-size: 10px; padding: 1px 7px; border-radius: 8px;
  background: #0f2718; color: #4ade80; border: 1px solid #4ade80;
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px;
}
.tx-badge-skipped {
  font-size: 10px; padding: 1px 7px; border-radius: 8px;
  background: #1e293b; color: #64748b; border: 1px solid #334155;
}

.tx-skip-body {
  font-size: 12px; color: #64748b; line-height: 1.55; margin: 0;
}

.tx-what {
  font-size: 12px; color: #94a3b8; line-height: 1.55; margin: 0;
}

/* ── Meta rows (trade-off, after this) ───────────────────────────────────── */
.tx-meta-row {
  display: flex;
  gap: 8px;
  font-size: 12px;
  line-height: 1.5;
}
.tx-meta-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #64748b;
  white-space: nowrap;
  padding-top: 1px;
  flex-shrink: 0;
  width: 64px;
}
.tx-meta-text { color: #94a3b8; }

.tx-ref {
  font-size: 11px; color: #60a5fa; text-decoration: none; align-self: flex-start;
}
.tx-ref:hover { text-decoration: underline; }

/* ── Computed outcome grid ───────────────────────────────────────────────── */
.tx-outcome {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 4px;
  overflow: hidden;
}

.tx-outcome-header,
.tx-outcome-row {
  display: grid;
  grid-template-columns: 72px 1fr 1fr 52px;
  gap: 0;
  padding: 5px 12px;
  align-items: center;
}

.tx-outcome-header {
  border-bottom: 1px solid #1e293b;
}

.tx-outcome-row {
  border-bottom: 1px solid #0f172a;
}
.tx-outcome-row:last-child { border-bottom: none; }

.tx-outcome-col-label {
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
}
.tx-outcome-col-label:nth-child(2),
.tx-outcome-col-label:nth-child(3) { text-align: right; }

.tx-outcome-stat-label {
  font-size: 10px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}
.tx-outcome-stat-before {
  font-size: 12px;
  font-family: ui-monospace, monospace;
  color: #64748b;
  text-align: right;
}
.tx-outcome-stat-after {
  font-size: 12px;
  font-family: ui-monospace, monospace;
  font-weight: 600;
  text-align: right;
}
.tx-improved { color: #4ade80; }
.tx-worse    { color: #f87171; }

.tx-reduction {
  font-size: 11px;
  font-weight: 700;
  text-align: right;
  display: block;
}
.tx-reduction--good { color: #4ade80; }
.tx-reduction--bad  { color: #f87171; }

/* ── Footer ────────────────────────────────────────────────────────────────── */
.tx-footer { font-size: 11px; color: #334155; font-style: italic; text-align: right; }

/* ── Light theme ─────────────────────────────────────────────────────────── */
:global(.theme-light) .tx-item           { border-color: #e2e8f0; }
:global(.theme-light) .tx-item--recommended { background: #f0fdf4; border-color: #86efac; }
:global(.theme-light) .tx-name           { color: #1e293b; }
:global(.theme-light) .tx-outcome        { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .tx-footer         { color: #94a3b8; }
</style>
