<!-- dsbf/dashboard/frontend/src/components/distributions/OutlierDetailCard.vue

  Per-column outlier breakdown from detect_outliers (unified task).

  Structure:
  - Consensus / interpretation banner
  - Method grid (IQR, Z-score, MAD, Isolation Forest as 4th row)
  - Notable values (top extreme values from MAD)

  IF is folded into the method grid as a 4th row. Its "flagged" count is
  dataset-wide, but we compute the overlap with this column's IQR row
  indices to give a per-column relevance signal.

  Props
  ─────
  column : String
  tasks  : Object
-->

<template>
  <div class="outlier-card">

    <div v-if="state === 'not_run'" class="es-not-run">
      Outlier analysis did not run for this column.
    </div>

    <div v-else-if="state === 'clean'" class="es-empty">
      ✓ No outliers detected by any method for this column.
    </div>

    <template v-else>

      <!-- ── Interpretation banner ──────────────────────────────────────── -->
      <div class="interp-banner" :class="`interp-banner--${interpLevel}`">
        <span class="interp-icon">{{ interpIcon }}</span>
        <div class="interp-body">
          <div class="interp-title">{{ interpTitle }}</div>
          <div class="interp-desc">{{ interpDesc }}</div>
        </div>
      </div>

      <!-- ── Method grid ────────────────────────────────────────────────── -->
      <div class="methods-grid">
        <div class="method-row method-row--header">
          <span>Method</span>
          <span class="col-right">Flagged</span>
          <span class="col-right">% of rows</span>
          <span>Notes</span>
        </div>

        <!-- IQR -->
        <div class="method-row">
          <span class="method-name">
            <span class="method-label-row">
              IQR
              <TooltipIcon
                text="Interquartile Range method. Flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. Simple and distribution-free, but sensitive to the spread of the middle 50% of data."
                direction="up" align="left"
              />
            </span>
            <span class="method-desc">1.5× fence</span>
          </span>
          <span class="method-count col-right" :class="iqrFlagged ? 'flagged-val' : 'clean-val'">
            {{ data.iqr?.outlier_count?.toLocaleString() ?? '-' }}
          </span>
          <span class="method-pct col-right">
            {{ data.iqr?.outlier_pct != null ? (data.iqr.outlier_pct * 100).toFixed(2) + '%' : '-' }}
          </span>
          <span class="method-note">
            <template v-if="data.iqr?.lower_fence != null">
              Fences: [{{ fmtN(data.iqr.lower_fence) }}, {{ fmtN(data.iqr.upper_fence) }}]
            </template>
          </span>
        </div>

        <!-- Z-score -->
        <div class="method-row">
          <span class="method-name">
            <span class="method-label-row">
              Z-score
              <TooltipIcon
                text="Standard score method. Flags values whose distance from the mean exceeds a threshold (default |z| > 3). Assumes approximate normality - less reliable on highly skewed distributions."
                direction="up" align="left"
              />
            </span>
            <span class="method-desc">|z| &gt; {{ data.zscore?.threshold ?? 3 }}</span>
          </span>
          <span class="method-count col-right" :class="zFlagged ? 'flagged-val' : 'clean-val'">
            {{ data.zscore?.outlier_count?.toLocaleString() ?? '-' }}
          </span>
          <span class="method-pct col-right">
            {{ data.zscore?.outlier_pct != null ? (data.zscore.outlier_pct * 100).toFixed(2) + '%' : '-' }}
          </span>
          <span class="method-note">
            <template v-if="data.zscore?.max_zscore != null">
              Max |z| = {{ fmtN(data.zscore.max_zscore) }}
            </template>
          </span>
        </div>

        <!-- MAD -->
        <div class="method-row">
          <span class="method-name">
            <span class="method-label-row">
              MAD
              <TooltipIcon
                text="Median Absolute Deviation method. A robust alternative to Z-score that uses the median instead of the mean. More reliable on skewed or non-normal distributions (Iglewicz & Hoaglin 1993)."
                direction="up" align="left"
              />
            </span>
            <span class="method-desc">modified z &gt; {{ data.mad?.threshold ?? 3.5 }}</span>
          </span>
          <span class="method-count col-right" :class="madFlagged ? 'flagged-val' : 'clean-val'">
            {{ data.mad?.outlier_count?.toLocaleString() ?? '-' }}
          </span>
          <span class="method-pct col-right">
            {{ data.mad?.outlier_pct != null ? (data.mad.outlier_pct * 100).toFixed(2) + '%' : '-' }}
          </span>
          <span class="method-note">
            <template v-if="data.mad?.max_modified_z_score != null">
              Max |MZ| = {{ fmtN(data.mad.max_modified_z_score) }}
            </template>
          </span>
        </div>

        <!-- Isolation Forest -->
        <div v-if="ifData" class="method-row">
          <span class="method-name">
            <span class="method-label-row">
              Isolation Forest
              <TooltipIcon
                text="A tree-based algorithm that detects multivariate anomalies - rows unusual across multiple features simultaneously. Runs dataset-wide, not per column. The overlap count shows how many of this column's IQR outlier rows were also flagged by IF, strengthening those signals."
                direction="up" align="left"
              />
            </span>
            <span class="method-desc">multivariate · dataset-wide</span>
          </span>
          <span class="method-count col-right" :class="ifOverlap > 0 ? 'flagged-val' : 'clean-val'">
            {{ ifOverlap.toLocaleString() }}
            <span class="method-note" style="display:inline;margin-left:4px;">overlap</span>
          </span>
          <span class="method-pct col-right">
            {{ ifData.n_flagged != null
              ? `${ifData.n_flagged.toLocaleString()} total`
              : '-' }}
          </span>
          <span class="method-note">
            {{ ifOverlap }} of {{ data.iqr?.outlier_count ?? 0 }} IQR rows also in IF
          </span>
        </div>

      </div>

      <!-- ── Notable values ─────────────────────────────────────────────── -->
      <div v-if="notableValues.length" class="notable-section">
        <div class="notable-label">
          Extreme values flagged
          <TooltipIcon
            text="The most extreme values identified by the MAD method. Reviewing these directly is the fastest way to determine whether outliers are data errors, rare-but-valid observations, or expected tail behavior."
            direction="up" align="left"
          />
        </div>
        <div class="notable-chips">
          <span
            v-for="(v, i) in notableValues"
            :key="i"
            class="notable-chip"
          >{{ fmtN(v) }}</span>
        </div>
      </div>

    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  column: { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
})

function fmtN(v) {
  if (v == null) return '-'
  const n = Number(v)
  if (!isFinite(n)) return '-'
  if (Math.abs(n) >= 10000) return n.toLocaleString(undefined, { maximumFractionDigits: 1 })
  if (Math.abs(n) >= 10)    return n.toFixed(2)
  return n.toPrecision(4).replace(/\.?0+$/, '')
}

// ── Data ─────────────────────────────────────────────────────────────────────

const data   = computed(() => props.tasks?.detect_outliers?.data?.[props.column] ?? null)
const ifData = computed(() =>
  props.tasks?.detect_outliers?.data?.['__dataset__']?.isolation_forest ?? null
)

// Overlap: IQR-flagged rows for this column that were also flagged by IF
const ifOverlap = computed(() => {
  const iqrRows = new Set(data.value?.iqr?.outlier_row_indices ?? [])
  if (!iqrRows.size || !ifData.value) return 0
  return (ifData.value.flagged_row_indices ?? []).filter(i => iqrRows.has(i)).length
})

const state = computed(() => {
  if (!data.value) return 'not_run'
  if ((data.value.methods_flagging?.length ?? 0) === 0) return 'clean'
  return 'ready'
})

const iqrFlagged = computed(() => (data.value?.iqr?.outlier_count ?? 0) > 0)
const zFlagged   = computed(() => (data.value?.zscore?.outlier_count ?? 0) > 0)
const madFlagged = computed(() => (data.value?.mad?.outlier_count ?? 0) > 0)

// Deduplicated top outlier values from MAD (most robust method)
const notableValues = computed(() => {
  const vals = data.value?.mad?.top_outlier_values ?? []
  return [...new Set(vals)].slice(0, 5)
})

// ── Interpretation ────────────────────────────────────────────────────────────

const interpLevel = computed(() => {
  if (!data.value) return 'none'
  const count  = data.value.methods_flagging?.length ?? 0
  const pct    = data.value.iqr?.outlier_pct ?? 0
  if (data.value.consensus && pct >= 0.05) return 'high'
  if (data.value.consensus)                return 'moderate'
  return 'low'
})

const interpIcon = computed(() =>
  ({ high: '⚠', moderate: 'ℹ', low: 'ℹ', none: '' }[interpLevel.value])
)

const interpTitle = computed(() => {
  const count = data.value?.methods_flagging?.length ?? 0
  if (data.value?.consensus) {
    const pct = (data.value.iqr?.outlier_pct ?? 0) * 100
    return pct >= 5
      ? `Strong outlier signal - ${pct.toFixed(1)}% of rows flagged by ${count} methods`
      : `Cross-method consensus - ${count} methods agree on outlier presence`
  }
  return count > 0
    ? `Weak outlier signal - flagged by ${count} of 3 methods`
    : 'No outliers detected'
})

const interpDesc = computed(() => {
  const consensus = data.value?.consensus ?? false
  const pct       = (data.value?.iqr?.outlier_pct ?? 0) * 100
  const skew      = props.tasks?.detect_skewness?.data?.[props.column]
  const hasSkew   = skew != null && Math.abs(skew) > 1

  if (!consensus) {
    return 'Only one method flagged values. This is common for columns with naturally heavy or asymmetric tails. ' +
           (hasSkew ? 'The column is already flagged as skewed, so extreme tail values are expected.' :
            'Review the extreme values below to decide if any warrant attention.')
  }
  if (pct < 1) {
    return 'A small proportion of rows are flagged by multiple methods. ' +
           'These may be genuine rare events, measurement errors, or data entry issues. ' +
           'Review the extreme values below to assess which is more likely.'
  }
  return `${pct.toFixed(1)}% of rows are flagged as outliers by multiple methods. ` +
         'This is a substantial proportion - consider whether the column has a naturally extreme distribution, ' +
         'or whether data quality issues may be inflating the tails.'
})
</script>

<style scoped>
/* ── Empty states ────────────────────────────────────────────────────────── */
.es-not-run { color: #64748b; font-size: 13px; padding: 8px 0; text-align: center; }
.es-empty   { color: #4ade80; font-size: 13px; padding: 8px 0; text-align: center; }

/* ── Interpretation banner ───────────────────────────────────────────────── */
.interp-banner {
  display: flex;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 6px;
  border: 1px solid;
  margin-bottom: 14px;
  align-items: flex-start;
}
.interp-banner--high     { background: #3d2a00; border-color: #fbbf24; }
.interp-banner--moderate { background: #1e293b; border-color: #334155; }
.interp-banner--low      { background: #1e293b; border-color: #334155; }

.interp-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
.interp-body { display: flex; flex-direction: column; gap: 4px; }

.interp-title {
  font-size: 13px;
  font-weight: 600;
}
.interp-banner--high     .interp-title { color: #fde68a; }
.interp-banner--moderate .interp-title { color: #94a3b8; }
.interp-banner--low      .interp-title { color: #94a3b8; }

.interp-desc {
  font-size: 12px;
  line-height: 1.55;
}
.interp-banner--high     .interp-desc { color: #fbbf24; }
.interp-banner--moderate .interp-desc { color: #64748b; }
.interp-banner--low      .interp-desc { color: #64748b; }

/* ── Methods grid ────────────────────────────────────────────────────────── */
.methods-grid { display: flex; flex-direction: column; }

.method-row {
  display: grid;
  grid-template-columns: 180px 90px 100px 1fr;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid #1e293b;
  font-size: 13px;
}
.method-row:last-child { border-bottom: none; }
.method-row:hover:not(.method-row--header) { background: rgba(255,255,255,0.02); }

.method-row--header {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
  border-bottom: 1px solid #334155;
  padding-bottom: 8px;
}
.col-right { text-align: right; }

.method-name {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-family: ui-monospace, monospace;
  font-size: 12px;
  font-weight: 600;
  color: #e2e8f0;
}
.method-label-row {
  display: flex;
  align-items: center;
  gap: 4px;
}
.method-desc {
  font-size: 10px;
  font-weight: 400;
  color: #64748b;
  font-family: inherit;
}

.method-count { font-size: 13px; font-weight: 600; }
.flagged-val  { color: #fbbf24; }
.clean-val    { color: #4ade80; }
.method-pct   { font-size: 12px; color: #64748b; }
.method-note  { font-size: 11px; color: #64748b; font-family: ui-monospace, monospace; }

/* ── Notable values ──────────────────────────────────────────────────────── */
.notable-section {
  margin-top: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.notable-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
  display: flex;
  align-items: center;
  gap: 4px;
}
.notable-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.notable-chip {
  font-family: ui-monospace, monospace;
  font-size: 12px;
  font-weight: 600;
  color: #fbbf24;
  background: #3d2a00;
  border: 1px solid #fbbf24;
  border-radius: 4px;
  padding: 3px 10px;
}

/* ── Light theme ─────────────────────────────────────────────────────────── */
:global(.theme-light) .interp-banner--high     { background: #fffbeb; border-color: #fcd34d; }
:global(.theme-light) .interp-banner--moderate { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .interp-banner--low      { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .interp-banner--high  .interp-title { color: #92400e; }
:global(.theme-light) .interp-banner--high  .interp-desc  { color: #b45309; }
:global(.theme-light) .method-row   { border-bottom-color: #f1f5f9; }
:global(.theme-light) .method-row--header { border-bottom-color: #e2e8f0; }
:global(.theme-light) .method-name  { color: #1e293b; }
:global(.theme-light) .notable-chip { background: #fffbeb; border-color: #fcd34d; color: #92400e; }
</style>
