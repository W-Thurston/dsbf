<template>
  <div class="distributions-tab">

    <!-- Left: column browser -->
    <div class="browser-pane card">
      <div class="card-title">Columns</div>
      <ColumnBrowser
        :tasks="tasks"
        :selected="selectedColumn"
        @select="onSelectColumn"
      />
    </div>

    <!-- Right: detail panel -->
    <div v-if="selectedColumn" class="detail-pane">

      <!-- Column heading -->
      <div class="col-heading">
        <span class="col-heading-name">{{ selectedColumn }}</span>
        <span class="col-heading-badge" :class="`intent-${columnIntent}`">{{ columnIntent }}</span>
        <span class="col-heading-dtype muted">{{ columnDtype }}</span>
      </div>

      <!-- ── 1. Stats strip - always visible, no collapse ─────────────────── -->
      <ColumnStatsStrip :column="selectedColumn" :tasks="tasks" />

      <!-- ── 2. Distribution plot - non-collapsible ──────────────────────── -->
      <div class="plot-guidance-row">
        <div class="plot-pane card">
          <div class="card-title-row">
            <div class="card-title">Distribution</div>
            <div v-if="columnIntent === 'continuous'" class="plot-toggle">
              <button
                v-for="opt in plotOptions"
                :key="opt.key"
                class="toggle-btn"
                :class="{ active: activePlotType === opt.key }"
                @click="activePlotType = opt.key"
              >{{ opt.label }}</button>
            </div>
          </div>
          <PlotCard
            v-if="columnFigure"
            :title="plotTitle"
            :figure="columnFigure"
            :theme="theme"
            :show-title="false"
          />
          <div v-else class="no-figure">
            <span class="no-figure-icon">📊</span>
            No figure available for this column.
          </div>
        </div>

        <!-- ── 3. Guidance - non-collapsible ────────────────────────────── -->
        <div class="guidance-pane">
          <ColumnGuidanceCard :column="selectedColumn" :tasks="tasks" />
        </div>
      </div>

      <!-- ── 3b. Bimodal callout (continuous only, when flagged) ──────────── -->
      <div
        v-if="columnIntent === 'continuous' && isBimodal"
        class="bimodal-callout"
      >
        <span class="bimodal-icon">⚠</span>
        <div>
          <strong>Possible bimodal distribution detected.</strong>
          A two-component mixture model fits this column significantly better than a single-component model
          (BIC improvement: {{ bimodalBicImprovement }}).
          This often indicates two distinct subpopulations in the data - e.g. two groups, time periods,
          or conditions that were recorded together. Analysing subgroups separately may be more informative
          than treating this as a single distribution.
        </div>
      </div>

      <!-- ── 4. Percentile breakdown / Value counts - collapsible ───────── -->
      <div class="dist-section card">
        <div class="dist-section-header" @click="toggleSection('context')">
          <span class="dist-section-title">
            {{ columnIntent === 'continuous' ? 'Percentile Breakdown' : 'Value Counts' }}
            <TooltipIcon
              :text="columnIntent === 'continuous'
                ? 'Full percentile distribution from min to max. Useful for understanding spread, identifying skew in the tails, and spotting outlier boundaries.'
                : 'Count and proportion of each value. Shows the frequency distribution for categorical, boolean, and datetime columns.'"
              direction="down" align="left"
            />
          </span>
          <button class="dist-collapse-btn">{{ openSections.has('context') ? '▲' : '▼' }}</button>
        </div>
        <Transition name="dist-expand">
          <div v-if="openSections.has('context')" class="dist-section-body">
            <PercentilesTable v-if="columnIntent === 'continuous'" :column="selectedColumn" :tasks="tasks" :embedded="true" />
            <template v-else>
              <ValueCountsTable :column="selectedColumn" :tasks="tasks" :embedded="true" />
              <TextLengthCard
                v-if="columnIntent === 'categorical' || columnIntent === 'text'"
                :column="selectedColumn"
                :tasks="tasks"
                style="margin-top: 14px;"
              />
            </template>
          </div>
        </Transition>
      </div>

      <!-- Continuous-only sections below this point -->
      <template v-if="columnIntent === 'continuous'">

        <!-- ── 5. Outlier analysis ─────────────────────────────────────── -->
        <div class="dist-section card">
          <div class="dist-section-header" @click="toggleSection('outliers')">
            <span class="dist-section-title">
              Outlier Analysis
              <TooltipIcon
                text="Runs three independent detection methods (IQR, Z-score, MAD) and checks for cross-method consensus. Consensus means at least two methods agree a value is unusual - a stronger signal than any single method alone."
                direction="down" align="left"
              />
            </span>
            <span class="dist-section-badge" :class="outlierBadgeClass">{{ outlierBadgeText }}</span>
            <button class="dist-collapse-btn">{{ openSections.has('outliers') ? '▲' : '▼' }}</button>
          </div>
          <Transition name="dist-expand">
            <div v-if="openSections.has('outliers')" class="dist-section-body">
              <OutlierDetailCard :column="selectedColumn" :tasks="tasks" :embedded="true" />
            </div>
          </Transition>
        </div>

        <!-- ── 6. Normality ────────────────────────────────────────────── -->
        <div class="dist-section card">
          <div class="dist-section-header" @click="toggleSection('normality')">
            <span class="dist-section-title">
              Normality
              <TooltipIcon
                text="Tests whether the column's distribution is consistent with a normal (Gaussian) distribution using the Kolmogorov-Smirnov and Jarque-Bera tests. The Q-Q plot shows how empirical quantiles compare to theoretical normal quantiles - points close to the diagonal indicate normality."
                direction="down" align="left"
              />
            </span>
            <span class="dist-section-badge" :class="normalityBadgeClass">{{ normalityBadgeText }}</span>
            <button class="dist-collapse-btn">{{ openSections.has('normality') ? '▲' : '▼' }}</button>
          </div>
          <Transition name="dist-expand">
            <div v-if="openSections.has('normality')" class="dist-section-body">
              <div v-if="normalityResult" class="normality-interp">
                <div class="normality-result-row">
                  <span class="normality-badge" :class="normalityResult === 'normal' ? 'badge-normal' : 'badge-nonnormal'">
                    {{ normalityResult === 'normal' ? 'Consistent with normal' : 'Non-normal' }}
                  </span>
                  <span v-if="normalityN" class="normality-n">n = {{ normalityN?.toLocaleString() }}</span>
                </div>
                <p class="normality-prose">{{ normalityProse }}</p>
                <div class="normality-pvalues" v-if="normalityKsPval != null || normalityJbPval != null">
                  <span v-if="normalityKsPval != null" class="normality-pval">
                    KS test: p = {{ normalityKsPval.toFixed(4) }}
                  </span>
                  <span v-if="normalityJbPval != null" class="normality-pval">
                    Jarque-Bera: p = {{ normalityJbPval.toFixed(4) }}
                  </span>
                </div>
                <div class="normality-placeholder">
                  Q-Q plot visualisation coming in the D3 refactor.
                </div>
              </div>
              <div v-else class="dist-not-run">
                Normality tests did not run for this column, or this column passed
                and was excluded from further analysis (Q-Q plots are only computed
                for non-normal columns by default).
              </div>
            </div>
          </Transition>
        </div>

        <!-- ── 8. Top correlations ─────────────────────────────────────── -->
        <div v-if="runKey" class="dist-section card">
          <div class="dist-section-header" @click="toggleSection('correlations')">
            <span class="dist-section-title">
              Top Correlations
              <TooltipIcon
                text="Pearson correlation between this column and other numeric columns. Values near ±1 indicate a strong linear relationship. High correlations (≥0.9) may signal data leakage or redundant features worth investigating."
                direction="down" align="left"
              />
            </span>
            <button class="dist-collapse-btn">{{ openSections.has('correlations') ? '▲' : '▼' }}</button>
          </div>
          <Transition name="dist-expand">
            <div v-if="openSections.has('correlations')" class="dist-section-body">
              <ColumnCorrelationsPanel
                :run-key="runKey"
                :column="selectedColumn"
                :tasks="tasks"
                @select-column="onSelectColumn"
              />
            </div>
          </Transition>
        </div>

      </template>
    </div>

    <!-- Empty state: no column selected -->
    <div v-else class="empty-state card">
      <span class="empty-icon">←</span>
      <p>Select a column from the browser to explore its distribution.</p>
    </div>

  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import TooltipIcon                from '../../components/TooltipIcon.vue'
import ColumnBrowser              from '../../components/distributions/ColumnBrowser.vue'
import ColumnStatsStrip           from '../../components/distributions/ColumnStatsStrip.vue'
import ColumnGuidanceCard         from '../../components/distributions/ColumnGuidanceCard.vue'
import PercentilesTable           from '../../components/distributions/PercentilesTable.vue'
import ValueCountsTable           from '../../components/distributions/ValueCountsTable.vue'
import TextLengthCard             from '../../components/distributions/TextLengthCard.vue'
import OutlierDetailCard          from '../../components/distributions/OutlierDetailCard.vue'
import ColumnCorrelationsPanel    from '../../components/distributions/ColumnCorrelationsPanel.vue'
import PlotCard                   from '../../components/overview/PlotCard.vue'
import { figureForColumn }        from '../../utils.js'

const props = defineProps({
  run:           { type: Object, default: null },
  tasks:         { type: Object, default: () => ({}) },
  figures:       { type: Array,  default: () => [] },
  theme:         { type: String, default: 'dark' },
})

const selectedColumn = ref(null)
const activePlotType = ref('histogram')
const openSections   = ref(new Set())


const plotOptions = [
  { key: 'histogram', label: 'Histogram' },
  { key: 'boxplot',   label: 'Box Plot'  },
]

function toggleSection(key) {
  const s = new Set(openSections.value)
  s.has(key) ? s.delete(key) : s.add(key)
  openSections.value = s
}

function onSelectColumn(col) {
  selectedColumn.value = col
  activePlotType.value  = 'histogram'

  const intent = props.tasks.infer_types?.data?.[col]?.analysis_intent_dtype ?? 'unknown'

  // Always open context (percentiles / value counts)
  const s = new Set(['context'])

  if (intent === 'continuous') {
    // Open outliers if anything was flagged
    const od = props.tasks?.detect_outliers?.data?.[col]
    if (od && (od.methods_flagging?.length ?? 0) > 0) s.add('outliers')

    // Open normality if non-normal result exists
    const nr = props.tasks?.normality_tests?.data?.[col]?.result
    if (nr === 'non_normal') s.add('normality')

    // Always open correlations
    s.add('correlations')
  }

  openSections.value = s
}

// ── Column metadata ──────────────────────────────────────────────────────────

const columnIntent = computed(() =>
  props.tasks.infer_types?.data?.[selectedColumn.value]?.analysis_intent_dtype ?? 'unknown'
)
const columnDtype = computed(() =>
  props.tasks.infer_types?.data?.[selectedColumn.value]?.inferred_dtype ?? ''
)
const runKey = computed(() => props.run?.run_key ?? '')

// ── Figure selection ─────────────────────────────────────────────────────────

const columnFigure = computed(() => {
  const col    = selectedColumn.value
  const intent = columnIntent.value
  if (!col || !props.figures?.length) return null
  if (intent === 'continuous') {
    const type = activePlotType.value
    return (
      figureForColumn(props.figures, col, type, 'interactive', props.theme) ??
      figureForColumn(props.figures, col, type, 'static',      props.theme) ??
      figureForColumn(props.figures, col, 'composite', 'static', props.theme) ??
      null
    )
  }
  return (
    figureForColumn(props.figures, col, 'bar', 'interactive', props.theme) ??
    figureForColumn(props.figures, col, 'bar', 'static',      props.theme) ??
    null
  )
})

const plotTitle = computed(() =>
  columnIntent.value === 'continuous'
    ? (activePlotType.value === 'boxplot' ? 'Box Plot' : 'Distribution')
    : 'Value Counts'
)

// ── Section badges and not-run states ────────────────────────────────────────

const outlierBadgeText = computed(() => {
  const d = props.tasks?.detect_outliers?.data?.[selectedColumn.value]
  if (!d) return 'Not computed'
  if ((d.methods_flagging?.length ?? 0) === 0) return 'Nothing flagged'
  return d.consensus ? 'Consensus flagging' : 'Weak flagging'
})
const outlierBadgeClass = computed(() => {
  const d = props.tasks?.detect_outliers?.data?.[selectedColumn.value]
  if (!d) return 'dist-badge--muted'
  if ((d.methods_flagging?.length ?? 0) === 0) return 'dist-badge--green'
  return d.consensus ? 'dist-badge--amber' : 'dist-badge--muted'
})

const normalityBadgeText = computed(() => {
  const result = props.tasks?.normality_tests?.data?.[selectedColumn.value]?.result
  if (!result) return 'Not computed'
  return result === 'normal' ? 'Consistent with normal' : 'Non-normal'
})
const normalityBadgeClass = computed(() => {
  const result = props.tasks?.normality_tests?.data?.[selectedColumn.value]?.result
  if (!result) return 'dist-badge--muted'
  return result === 'normal' ? 'dist-badge--green' : 'dist-badge--amber'
})

// ── Bimodal ───────────────────────────────────────────────────────────────────

const isBimodal = computed(() =>
  props.tasks?.detect_bimodal_distribution?.data?.bimodal_flags?.[selectedColumn.value] === true
)

const bimodalBicImprovement = computed(() => {
  const scores = props.tasks?.detect_bimodal_distribution?.data?.bic_scores?.[selectedColumn.value]
  if (!scores?.relative_improvement) return null
  return `${(scores.relative_improvement * 100).toFixed(1)}%`
})

// ── Normality prose ───────────────────────────────────────────────────────────

const normalityData = computed(() =>
  props.tasks?.normality_tests?.data?.[selectedColumn.value] ?? null
)

const normalityResult = computed(() => normalityData.value?.result ?? null)
const normalityN      = computed(() => normalityData.value?.n ?? null)
const normalityKsPval = computed(() =>
  normalityData.value?.ks_pvalue ?? normalityData.value?.ks_normal_p ?? null
)
const normalityJbPval = computed(() =>
  normalityData.value?.jb_pvalue ?? normalityData.value?.jb_p ?? null
)

const normalityProse = computed(() => {
  const result = normalityResult.value
  const skew   = props.tasks?.detect_skewness?.data?.[selectedColumn.value]
  if (!result) return ''
  if (result === 'normal') {
    return 'This column is statistically consistent with a normal distribution. ' +
           'Many parametric methods (linear regression, t-tests, Pearson correlation) ' +
           'assume or benefit from normality, so this column is well-suited for those approaches.'
  }
  // non-normal
  const hasSkew = skew != null && Math.abs(skew) > 1
  if (hasSkew) {
    return `This column is non-normal, which is consistent with its skewness (${skew > 0 ? 'right' : 'left'} tail). ` +
           'Parametric methods that assume normality may give unreliable results. ' +
           'Consider robust alternatives (e.g. Spearman correlation, Mann-Whitney U) ' +
           'or note this when interpreting any parametric analyses.'
  }
  return 'This column does not follow a normal distribution. ' +
         'Parametric methods that assume normality may give unreliable results - ' +
         'consider non-parametric alternatives or check whether a transformation ' +
         'would bring the distribution closer to normal before modelling.'
})
</script>

<style scoped>
.distributions-tab {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}

/* ── Browser pane ────────────────────────────────────────────────────────── */
.browser-pane {
  width: 220px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  max-height: calc(100vh - 200px);
  overflow: hidden;
  position: sticky;
  top: 16px;
}

/* ── Detail pane ─────────────────────────────────────────────────────────── */
.detail-pane {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* ── Column heading ──────────────────────────────────────────────────────── */
.col-heading {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  color: #f1f5f9;
  flex-wrap: wrap;
  padding: 4px 0;
}
.col-heading-name { font-family: monospace; font-size: 18px; }
.col-heading-badge {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.5px; padding: 3px 9px; border-radius: 9px; border: 1px solid;
}
.intent-continuous  { background: #1e3a5f; color: #60a5fa; border-color: #60a5fa; }
.intent-categorical { background: #14291f; color: #4ade80; border-color: #4ade80; }
.intent-boolean     { background: #2d1b4e; color: #c084fc; border-color: #c084fc; }
.intent-datetime    { background: #3d2510; color: #fb923c; border-color: #fb923c; }
.intent-text        { background: #3d0f29; color: #f472b6; border-color: #f472b6; }
.intent-unknown,
.intent-id          { background: #1e293b; color: #94a3b8; border-color: #64748b; }
.col-heading-dtype  { font-size: 13px; font-family: monospace; color: #64748b; }

/* ── Plot + guidance row (non-collapsible) ───────────────────────────────── */
.plot-guidance-row {
  display: flex;
  gap: 16px;
  align-items: stretch;
  height: 460px;
}
.plot-pane {
  flex: 3 1 0;
  min-width: 0;
  height: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.guidance-pane {
  flex: 1 1 0;
  min-width: 200px;
  height: 100%;
  overflow: hidden;
}

.plot-pane .card-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  margin-bottom: 12px;
}
.plot-pane .card-title-row .card-title { margin-bottom: 0; }
.plot-pane :deep(.plot-card) { flex: 1; min-height: 0; }
.plot-pane :deep(.plot-container) { min-height: 0; height: calc(100% - 32px); }
.guidance-pane :deep(.guidance-card) { height: 100%; overflow-y: auto; }

.no-figure {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: #64748b;
  font-size: 13px;
  background: #0f172a;
  border: 1px dashed #334155;
  border-radius: 8px;
}
.no-figure-icon { font-size: 28px; opacity: 0.4; }

/* ── Plot type toggle ────────────────────────────────────────────────────── */
.plot-toggle { display: flex; gap: 4px; }
.toggle-btn {
  padding: 3px 12px; border-radius: 6px; border: 1px solid #334155;
  background: #0f172a; color: #64748b; font-size: 11px;
  cursor: pointer; transition: all 0.15s; white-space: nowrap;
}
.toggle-btn:hover  { background: #1e293b; color: #94a3b8; }
.toggle-btn.active { background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }

/* ── Collapsible section cards ───────────────────────────────────────────── */
.dist-section { padding: 0; overflow: visible; }

.dist-section-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 20px;
  cursor: pointer;
  user-select: none;
  transition: background 0.12s;
}
.dist-section-header:hover { background: rgba(255,255,255,0.03); }

.dist-section-title {
  font-size: 13px;
  font-weight: 600;
  color: #e2e8f0;
  flex: 1;
  display: flex;
  align-items: center;
  gap: 4px;
}

.dist-section-badge {
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 9px;
  border: 1px solid;
  white-space: nowrap;
  flex-shrink: 0;
}
.dist-badge--green { background: #0f2718; color: #4ade80; border-color: #4ade80; }
.dist-badge--amber { background: #3d2a00; color: #fbbf24; border-color: #fbbf24; }
.dist-badge--muted { background: #1e293b; color: #64748b; border-color: #334155; }

.dist-collapse-btn {
  padding: 3px 8px; font-size: 11px; background: none;
  border: 1px solid #334155; border-radius: 4px; color: #64748b;
  cursor: pointer; transition: all 0.12s; flex-shrink: 0;
}
.dist-collapse-btn:hover { border-color: #60a5fa; color: #93c5fd; }

.dist-section-body {
  border-top: 1px solid #1e293b;
  padding: 16px 20px;
}

/* Strip card shell from embedded components */
.dist-section-body :deep(.context-card),
.dist-section-body :deep(.outlier-card),
.dist-section-body :deep(.qq-card),
.dist-section-body :deep(.tx-card),
.dist-section-body :deep(.correlations-card),
.dist-section-body :deep(.text-length-card) {
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
  box-shadow: none;
}
.dist-section-body :deep(.card-title) { display: none; }

/* ── Not-run explanation ─────────────────────────────────────────────────── */
.dist-not-run {
  font-size: 13px;
  color: #64748b;
  line-height: 1.6;
  padding: 4px 0 8px;
}

/* ── Bimodal callout ─────────────────────────────────────────────────────── */
.bimodal-callout {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 16px;
  background: #1e293b;
  border: 1px solid #60a5fa;
  border-radius: 8px;
  font-size: 13px;
  color: #94a3b8;
  line-height: 1.6;
}
.bimodal-callout strong { color: #bfdbfe; }
.bimodal-icon { font-size: 16px; flex-shrink: 0; margin-top: 2px; color: #60a5fa; }

/* ── Normality prose ─────────────────────────────────────────────────────── */
.normality-interp {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.normality-result-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.normality-badge {
  font-size: 12px;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 9px;
  border: 1px solid;
}
.badge-normal    { background: #0f2718; color: #4ade80; border-color: #4ade80; }
.badge-nonnormal { background: #3d2a00; color: #fbbf24; border-color: #fbbf24; }
.normality-n     { font-size: 12px; color: #64748b; }

.normality-prose {
  font-size: 13px;
  color: #94a3b8;
  line-height: 1.6;
  margin: 0;
}
.normality-pvalues {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}
.normality-pval {
  font-size: 12px;
  font-family: ui-monospace, monospace;
  color: #64748b;
}
.normality-placeholder {
  font-size: 12px;
  color: #334155;
  font-style: italic;
  padding: 8px 12px;
  background: #0f172a;
  border: 1px dashed #334155;
  border-radius: 4px;
}

/* ── Expand transition ───────────────────────────────────────────────────── */
.dist-expand-enter-active,
.dist-expand-leave-active {
  transition: opacity 0.18s, max-height 0.22s ease;
  max-height: 1400px;
  overflow: hidden;
}
.dist-expand-enter-from,
.dist-expand-leave-to { opacity: 0; max-height: 0; }

/* ── Empty state ─────────────────────────────────────────────────────────── */
.empty-state {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 12px; color: #94a3b8; min-height: 300px;
}
.empty-icon { font-size: 32px; }
.empty-state p { font-size: 14px; text-align: center; margin: 0; }

/* ── Light theme ─────────────────────────────────────────────────────────── */
:global(.theme-light) .dist-section-header:hover { background: rgba(0,0,0,0.02); }
:global(.theme-light) .dist-section-body         { border-top-color: #e2e8f0; }
:global(.theme-light) .dist-section-title        { color: #1e293b; }
:global(.theme-light) .col-heading               { color: #1e293b; }
:global(.theme-light) .no-figure                 { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .dist-not-run              { color: #94a3b8; }
:global(.theme-light) .bimodal-callout           { background: #eff6ff; border-color: #93c5fd; }
:global(.theme-light) .bimodal-callout strong    { color: #1d4ed8; }
:global(.theme-light) .bimodal-icon              { color: #2563eb; }
:global(.theme-light) .normality-prose           { color: #64748b; }
:global(.theme-light) .normality-pval            { color: #94a3b8; }
:global(.theme-light) .normality-placeholder     { background: #f8fafc; border-color: #e2e8f0; color: #94a3b8; }
</style>
