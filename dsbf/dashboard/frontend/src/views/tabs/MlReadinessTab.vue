<!-- dsbf/dashboard/frontend/src/views/tabs/MlReadinessTab.vue

  ML Readiness tab - surfaces the ml_readiness_scorer output.

  Mirrors the QualityTab layout but framed around preparation actions:
    1. Gate banner   - overall readiness gate + dimension summary cards
    2. Dimension sections - collapsible findings per preparation dimension,
                            sortable by column/level, 10-row scroll cap
    3. Clean features - columns with no ML findings

  Data source: GET /api/runs/{run_key}/ml-readiness
  Props: runKey (String, required)
-->

<template>
  <div class="ml-tab">

    <!-- ── Loading skeleton ───────────────────────────────────────────────── -->
    <div v-if="loading" class="ml-loading">
      <div class="ml-sk-banner" />
      <div class="ml-sk-grid">
        <div class="ml-sk-card" v-for="n in 5" :key="n" />
      </div>
      <div class="ml-sk-section" v-for="n in 3" :key="n" />
    </div>

    <!-- ── Error ──────────────────────────────────────────────────────────── -->
    <div v-else-if="error" class="ml-error card">{{ error }}</div>

    <!-- ── Unavailable ────────────────────────────────────────────────────── -->
    <div v-else-if="!available" class="ml-unavailable card">
      <div class="ml-unavail-icon">🔬</div>
      <div class="ml-unavail-title">ML Readiness analysis unavailable</div>
      <div class="ml-unavail-body">
        This run was profiled before the ML readiness scorer was introduced,
        or the scorer was skipped due to a failed dependency. Re-run the
        profiler to generate ML readiness findings.
      </div>
    </div>

    <template v-else>

      <!-- ── 1. Gate banner ──────────────────────────────────────────────── -->
      <div class="ml-banner card" :class="`ml-banner--${gateColor}`">
        <div class="ml-banner-left">
          <div class="ml-gate-badge" :class="`ml-gate--${data.readiness_gate}`">
            {{ gateLabel }}
          </div>
          <div class="ml-gate-desc">{{ gateDescription }}</div>
        </div>
        <div class="ml-banner-stats">
          <div class="ml-stat">
            <div class="ml-stat-value">{{ data.total_columns }}</div>
            <div class="ml-stat-label">Total columns</div>
          </div>
          <div class="ml-stat">
            <div class="ml-stat-value ml-text--good">{{ data.clean_columns?.length ?? 0 }}</div>
            <div class="ml-stat-label">Ready to use</div>
          </div>
          <div class="ml-stat">
            <div class="ml-stat-value ml-text--error">{{ errorDimCount }}</div>
            <div class="ml-stat-label">Dimensions blocking</div>
          </div>
        </div>
      </div>

      <!-- ── 2. Dimension summary cards ─────────────────────────────────── -->
      <div class="ml-summary-grid">
        <button
          v-for="dim in dimensions"
          :key="dim.key"
          class="ml-summary-card"
          :class="`ml-summary-card--${dim.level}`"
          @click="scrollTo(dim.key)"
        >
          <div class="ml-sc-header">
            <span class="ml-dot" :class="`ml-dot--${dim.level}`" />
            <span class="ml-sc-label">{{ dim.label }}</span>
          </div>
          <div
            class="ml-sc-count"
            :class="dim.affectedCount === 0 ? 'ml-text--good' : `ml-text--${dim.level}`"
          >
            {{ dim.affectedCount === 0
              ? 'All clear'
              : `${dim.affectedCount} col${dim.affectedCount === 1 ? '' : 's'}` }}
          </div>
          <div class="ml-sc-pct">
            {{ dim.affectedCount === 0
              ? 'No action needed'
              : `${(dim.pctAffected * 100).toFixed(1)}% of dataset` }}
          </div>
          <!-- Issue type counts -->
          <div v-if="dim.issueCounts.length" class="ml-sc-preview">
            <span
              v-for="ic in dim.issueCounts"
              :key="ic.label"
              class="ml-sc-chip"
            >{{ ic.label }}: {{ ic.count }}</span>
          </div>
        </button>
      </div>

      <!-- ── 3. Dimension sections ───────────────────────────────────────── -->
      <div
        v-for="dim in dimensions"
        :key="dim.key"
        :ref="el => sectionRefs[dim.key] = el"
        class="ml-section card"
      >
        <div class="ml-section-header" @click="toggleSection(dim.key)">
          <div class="ml-section-title">
            <span class="ml-dot" :class="`ml-dot--${dim.level}`" />
            <span>{{ dim.label }}</span>
            <TooltipIcon
              v-if="DIM_TOOLTIPS[dim.key]"
              :text="DIM_TOOLTIPS[dim.key]"
              direction="down"
              align="left"
            />
            <span class="ml-section-count" :class="`ml-text--${dim.level}`">
              {{ dim.affectedCount === 0
                ? 'All clear'
                : `${dim.affectedCount} col${dim.affectedCount === 1 ? '' : 's'}` +
                  (dim.findings.length !== dim.affectedCount
                    ? ` · ${dim.findings.length} finding${dim.findings.length === 1 ? '' : 's'}`
                    : '') }}
            </span>
          </div>
          <div class="ml-section-controls" @click.stop>
            <!-- Sort controls: only visible when section is open AND has findings -->
            <template v-if="dim.affectedCount > 0 && openSections.has(dim.key)">
              <span class="ml-sort-label">Sort:</span>
              <button
                v-for="opt in sortOptions"
                :key="opt.key"
                class="ml-sort-btn"
                :class="{ active: sortState[dim.key]?.by === opt.key }"
                @click="setSort(dim.key, opt.key)"
              >
                {{ opt.label }}
                <span v-if="sortState[dim.key]?.by === opt.key" class="ml-sort-arrow">
                  {{ sortState[dim.key].dir === 'asc' ? '↑' : '↓' }}
                </span>
              </button>
            </template>
            <button class="ml-collapse-btn" @click="toggleSection(dim.key)">
              {{ openSections.has(dim.key) ? '▲' : '▼' }}
            </button>
          </div>
        </div>

        <Transition name="ml-expand">
          <div v-if="openSections.has(dim.key)" class="ml-section-body">

            <div v-if="dim.affectedCount === 0" class="ml-all-clear">
              ✅ No preparation needed for this dimension.
            </div>

            <template v-else>
              <div class="ml-findings-table">
                <div class="ml-findings-header">
                  <span></span>
                  <span class="ml-col-col">Column</span>
                  <span class="ml-col-task">Source</span>
                  <span class="ml-col-finding">Finding</span>
                  <span class="ml-col-sev">Level</span>
                  <span></span>
                </div>
                <div class="ml-findings-scroll">
                  <div
                    v-for="(finding, i) in sortedFindings(dim)"
                    :key="i"
                    class="ml-finding-row"
                    :class="[
                      `ml-finding-row--${finding.level}`,
                      { 'ml-finding-row--expanded': expanded.has(`${dim.key}:${i}`),
                        'ml-finding-row--highlighted': highlighted.has(`${dim.key}:${i}`) }
                    ]"
                    @click="toggleExpanded(dim.key, i)"
                  >
                    <!-- Collapsed row -->
                    <button
                      class="ml-mark-btn"
                      :class="{ active: highlighted.has(`${dim.key}:${i}`) }"
                      :title="highlighted.has(`${dim.key}:${i}`) ? 'Marked as reviewed' : 'Mark as reviewed'"
                      @click.stop="toggleHighlight(dim.key, i)"
                    >{{ highlighted.has(`${dim.key}:${i}`) ? '★' : '☆' }}</button>
                    <span class="ml-col-col ml-finding-col">{{ finding.column }}</span>
                    <span class="ml-col-task ml-finding-task">{{ taskLabel(finding.task) }}</span>
                    <span class="ml-col-finding ml-finding-title">{{ finding.title }}</span>
                    <span class="ml-col-sev">
                      <span class="ml-sev-badge" :class="`ml-sev--${finding.level}`">
                        {{ finding.level }}
                      </span>
                    </span>
                    <span class="ml-expand-hint">{{ expanded.has(`${dim.key}:${i}`) ? '▴' : '▾' }}</span>

                    <!-- Expanded body + actions (spans full width) -->
                    <Transition name="ml-detail-expand">
                      <div
                        v-if="expanded.has(`${dim.key}:${i}`)"
                        class="ml-finding-detail"
                        @click.stop
                      >
                        <p class="ml-finding-body">{{ finding.body }}</p>

                        <!-- Metric stat strip (elevated from pills) -->
                        <div
                          v-if="finding.metric && Object.keys(finding.metric).length"
                          class="ml-metric-strip"
                        >
                          <div
                            v-for="(val, key) in finding.metric"
                            :key="key"
                            class="ml-metric-stat"
                          >
                            <div class="ml-metric-stat-label">{{ METRIC_LABELS[key] ?? key.replace(/_/g, ' ') }}</div>
                            <div class="ml-metric-stat-value">{{ formatMetricValue(key, val) }}</div>
                          </div>
                        </div>

                        <!-- Action list — full metadata from ACTION_META -->
                        <div
                          v-if="!hasTransformPreview(finding) && finding.actions?.length"
                          class="ml-action-list"
                        >
                          <div class="ml-action-list-label">Suggested actions</div>
                          <div
                            v-for="(act, j) in finding.actions"
                            :key="j"
                            class="ml-action-row"
                          >
                            <div class="ml-action-row-header">
                              <span class="ml-action-method">{{ act.method || act.action }}</span>
                              <span v-if="act.condition" class="ml-action-condition">{{ act.condition }}</span>
                            </div>
                            <template v-if="actionMeta(act)">
                              <p class="ml-action-what">{{ actionMeta(act).what }}</p>
                              <div class="ml-action-tradeoffs">
                                <span class="ml-tradeoff-label">Trade-off</span>
                                <span class="ml-tradeoff-text">{{ actionMeta(act).tradeoff }}</span>
                              </div>
                              <div class="ml-action-after">
                                <span class="ml-after-label">After this</span>
                                <span class="ml-after-text">{{ actionMeta(act).after }}</span>
                              </div>
                              <a
                                v-if="actionMeta(act).ref"
                                :href="actionMeta(act).ref.url"
                                target="_blank"
                                rel="noopener noreferrer"
                                class="ml-action-ref"
                              >↗ {{ actionMeta(act).ref.label }}</a>
                            </template>
                            <p v-else-if="act.detail" class="ml-action-what">{{ act.detail }}</p>
                          </div>
                        </div>

                        <!-- Transformation preview for skewness findings -->
                        <TransformationPreviewCard
                          v-if="dim.key === 'transformations' && hasTransformPreview(finding)"
                          :column="finding.column"
                          :tasks="tasks"
                          :embedded="true"
                        />

                        <!-- Additional suggestions not covered by the preview card -->
                        <div
                          v-if="hasTransformPreview(finding) && nonOverlappingActions(finding).length"
                          class="ml-action-list ml-action-list--additional"
                        >
                          <div class="ml-action-list-label">Additional suggestions to consider</div>
                          <div
                            v-for="(act, j) in nonOverlappingActions(finding)"
                            :key="j"
                            class="ml-action-row"
                          >
                            <div class="ml-action-row-header">
                              <span class="ml-action-method">{{ act.method || act.action }}</span>
                              <span v-if="act.condition" class="ml-action-condition">{{ act.condition }}</span>
                            </div>
                            <template v-if="actionMeta(act)">
                              <p class="ml-action-what">{{ actionMeta(act).what }}</p>
                              <div class="ml-action-tradeoffs">
                                <span class="ml-tradeoff-label">Trade-off</span>
                                <span class="ml-tradeoff-text">{{ actionMeta(act).tradeoff }}</span>
                              </div>
                              <div class="ml-action-after">
                                <span class="ml-after-label">After this</span>
                                <span class="ml-after-text">{{ actionMeta(act).after }}</span>
                              </div>
                              <a
                                v-if="actionMeta(act).ref"
                                :href="actionMeta(act).ref.url"
                                target="_blank"
                                rel="noopener noreferrer"
                                class="ml-action-ref"
                              >↗ {{ actionMeta(act).ref.label }}</a>
                            </template>
                            <p v-else-if="act.detail" class="ml-action-what">{{ act.detail }}</p>
                          </div>
                        </div>

                      </div>
                    </Transition>
                  </div>
                </div>
              </div>
            </template>

          </div>
        </Transition>
      </div>

      <!-- ── 4. Clean features ───────────────────────────────────────────── -->
      <div class="ml-section card">
        <div class="ml-section-header" @click="toggleSection('__clean__')">
          <div class="ml-section-title">
            <span class="ml-dot ml-dot--good" />
            <span>Ready Features</span>
            <span class="ml-section-count ml-text--good">
              {{ data.clean_columns?.length ?? 0 }} columns
            </span>
          </div>
          <button class="ml-collapse-btn">
            {{ openSections.has('__clean__') ? '▲' : '▼' }}
          </button>
        </div>
        <Transition name="ml-expand">
          <div v-if="openSections.has('__clean__')" class="ml-section-body">
            <div v-if="!data.clean_columns?.length" class="ml-all-clear">
              All columns have at least one ML finding.
            </div>
            <div v-else class="ml-clean-scroll">
              <div class="ml-clean-grid">
                <span
                  v-for="col in sortedCleanColumns"
                  :key="col"
                  class="ml-clean-chip"
                  :class="{ 'ml-clean-chip--highlighted': highlighted.has(`__clean__:${col}`) }"
                  @click="toggleHighlight('__clean__', col)"
                >{{ col }}</span>
              </div>
            </div>
          </div>
        </Transition>
      </div>

    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { getMlReadiness } from '../../api.js'
import TooltipIcon from '../../components/TooltipIcon.vue'
import TransformationPreviewCard from '../../components/distributions/TransformationPreviewCard.vue'
import { ACTION_META, actionMeta, normalizeMethod } from '../../utils/actionMeta.js'

const props = defineProps({
  runKey: { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
})

// ── State ─────────────────────────────────────────────────────────────────────

const loading   = ref(true)
const error     = ref(null)
const available = ref(false)
const data      = ref({})

const openSections = ref(new Set())
const sortState = ref({
  transformations: { by: 'level', dir: 'desc' },
  encoding:        { by: 'level', dir: 'desc' },
  missingness:     { by: 'level', dir: 'desc' },
  leakage:         { by: 'level', dir: 'desc' },
  unusable:        { by: 'level', dir: 'desc' },
})
const expanded    = ref(new Set())   // "dimKey:index" - expanded finding rows
const highlighted = ref(new Set())  // "dimKey:index" or "__clean__:col" - reviewed

const sectionRefs = ref({})

// ── Fetch ─────────────────────────────────────────────────────────────────────

async function fetchData(runKey) {
  loading.value   = true
  error.value     = null
  available.value = false
  data.value      = {}

  try {
    const result = await getMlReadiness(runKey)
    available.value = result.available ?? false
    if (available.value) data.value = result
  } catch (e) {
    error.value = `Failed to load ML readiness data: ${e.message}`
  } finally {
    loading.value = false
  }
}

onMounted(() => fetchData(props.runKey))
watch(() => props.runKey, key => { if (key) fetchData(key) })

// Auto-open sections with error findings; collapse all-clear ones
watch(data, (d) => {
  const cats = d.categories ?? {}
  const s = new Set()
  for (const [key, cat] of Object.entries(cats)) {
    const hasError = (cat.findings ?? []).some(f => f.level === 'error')
    const hasWarn  = (cat.findings ?? []).some(f => f.level === 'warn')
    if (hasError || hasWarn) s.add(key)
  }
  openSections.value = s
}, { immediate: false })

// ── Gate ──────────────────────────────────────────────────────────────────────

const GATE_COLOR = { ready: 'good', needs_work: 'warn', not_ready: 'error' }

const gateColor = computed(() => GATE_COLOR[data.value.readiness_gate] ?? 'warn')

const gateLabel = computed(() => ({
  ready:      '✓  Ready for Modeling',
  needs_work: '⚠  Needs Work Before Modeling',
  not_ready:  '✕  Not Ready for Modeling',
}[data.value.readiness_gate] ?? '-'))

const gateDescription = computed(() => ({
  ready:      'No preparation issues detected across any dimension. This dataset appears ready for most modeling workflows.',
  needs_work: 'Warnings present in one or more dimensions. Addressing the highlighted columns is likely to improve model reliability.',
  not_ready:  'Error-level issues present that are likely to cause problems for most modeling approaches. These are worth addressing before proceeding.',
}[data.value.readiness_gate] ?? ''))

const errorDimCount = computed(() =>
  Object.values(data.value.categories ?? {}).filter(c => c.level === 'red').length
)

// ── Dimensions ────────────────────────────────────────────────────────────────

const SEVERITY_RANK = { error: 3, warn: 2, info: 1, good: 0 }

const ISSUE_LABELS = {
  // Derive short labels from finding titles where possible,
  // otherwise fall back to level grouping
}

const dimensions = computed(() => {
  const cats = data.value.categories ?? {}
  return Object.entries(cats).map(([key, cat]) => {
    // Count by level for issue chips
    const levelMap = {}
    for (const f of cat.findings ?? []) {
      const lbl = f.level
      levelMap[lbl] = (levelMap[lbl] ?? 0) + 1
    }
    const issueCounts = Object.entries(levelMap)
      .sort((a, b) => (SEVERITY_RANK[b[0]] ?? 0) - (SEVERITY_RANK[a[0]] ?? 0))
      .map(([label, count]) => ({ label, count }))

    return {
      key,
      label:         cat.label,
      level:         cat.level,
      affectedCount: cat.affected_count,
      pctAffected:   cat.pct_affected,
      findings:      cat.findings ?? [],
      issueCounts,
    }
  })
})

// ── Sort options ──────────────────────────────────────────────────────────────

const sortOptions = [
  { key: 'column', label: 'Column'  },
  { key: 'level',  label: 'Level'   },
  { key: 'source', label: 'Source'  },
]

function setSort(dimKey, by) {
  const cur = sortState.value[dimKey]
  sortState.value = {
    ...sortState.value,
    [dimKey]: {
      by,
      dir: cur.by === by ? (cur.dir === 'asc' ? 'desc' : 'asc') : (by === 'level' ? 'desc' : 'asc'),
    },
  }
}

function sortedFindings(dim) {
  const { by, dir } = sortState.value[dim.key] ?? { by: 'level', dir: 'desc' }
  const mult = dir === 'asc' ? 1 : -1
  return [...dim.findings].sort((a, b) => {
    if (by === 'level')  return mult * ((SEVERITY_RANK[a.level] ?? 0) - (SEVERITY_RANK[b.level] ?? 0))
    if (by === 'column') return mult * a.column.localeCompare(b.column)
    if (by === 'source') return mult * a.task.localeCompare(b.task)
    return 0
  })
}

// ── Clean columns ─────────────────────────────────────────────────────────────

const sortedCleanColumns = computed(() =>
  [...(data.value.clean_columns ?? [])].sort()
)

// Dimension tooltips — describe source tasks, not general definitions
const DIM_TOOLTIPS = {
  transformations: 'Findings from skewness detection, outlier analysis, and bimodal distribution checks. Flags columns whose distribution shape may affect model performance.',
  encoding:        'Findings from categorical encoding suggestions and high-cardinality detection. Flags columns that need to be converted from strings to numeric representations before modeling.',
  missingness:     'Findings from null analysis. Flags columns where missing values are likely to cause errors or biased estimates in most modeling frameworks.',
  leakage:         'Findings from near-perfect correlation detection and duplicate column analysis. Flags columns that appear to encode the same information or may cause data leakage.',
  unusable:        'Findings from constant column, ID column, and dominant value detection. Flags columns that carry no signal and should be excluded from any model.',
}

// ── Task label helper ─────────────────────────────────────────────────────────

const TASK_SHORT = {
  detect_skewness:              'Skewness',
  detect_outliers:              'Outliers',
  suggest_numerical_binning:    'Binning',
  detect_bimodal_distribution:  'Bimodal',
  detect_high_cardinality:      'Cardinality',
  suggest_categorical_encoding: 'Encoding',
  summarize_boolean_fields:     'Boolean',
  summarize_nulls:              'Nulls',
  summarize_numeric:            'Numeric',
  detect_data_leakage:          'Leakage',
  detect_constant_columns:      'Constant',
  detect_id_columns:            'ID',
  detect_single_dominant_value: 'Dominant',
  detect_zeros:                 'Zeros',
}

function taskLabel(taskName) {
  return TASK_SHORT[taskName] ?? taskName.replace('detect_', '').replace(/_/g, ' ')
}

// ── Metric stat strip helpers ─────────────────────────────────────────────────

const METRIC_LABELS = {
  skewness:          'Skewness',
  mean:              'Mean',
  median:            'Median',
  std:               'Std Dev',
  null_pct:          'Null %',
  null_count:        'Null count',
  n_rows:            'Total rows',
  correlation:       'Correlation',
  correlated_with:   'Correlated with',
  threshold:         'Threshold',
  cardinality:       'Unique values',
  suggested_encoding:'Suggested encoding',
  n_unique:          'Unique values',
  vif_score:         'VIF score',
}

function formatMetricValue(key, val) {
  if (val == null) return '—'
  if (typeof val === 'number') {
    if (key === 'null_pct') return `${(val * 100).toFixed(1)}%`
    if (key === 'correlation') return val.toFixed(4)
    if (Number.isInteger(val)) return val.toLocaleString()
    return val.toPrecision(4).replace(/\.?0+$/, '')
  }
  return String(val)
}

// ── Transform preview helpers ─────────────────────────────────────────────────

function hasTransformPreview(finding) {
  return finding.task === 'detect_skewness' &&
    !!(props.tasks?.transformation_preview?.data?.[finding.column])
}

function nonOverlappingActions(finding) {
  const all = finding.actions ?? []
  if (!hasTransformPreview(finding)) return all
  const previewKeys = new Set(
    Object.keys(
      props.tasks?.transformation_preview?.data?.[finding.column]?.transforms ?? {}
    ).map(normalizeMethod)
  )
  return all.filter(act => {
    const method = normalizeMethod(act.method ?? act.action)
    return !previewKeys.has(method)
  })
}

// ── Interactions ──────────────────────────────────────────────────────────────

function toggleSection(key) {
  const s = new Set(openSections.value)
  s.has(key) ? s.delete(key) : s.add(key)
  openSections.value = s
}

function toggleExpanded(dimKey, i) {
  const key = `${dimKey}:${i}`
  const s = new Set(expanded.value)
  s.has(key) ? s.delete(key) : s.add(key)
  expanded.value = s
}

function toggleHighlight(dimKey, id) {
  const key = `${dimKey}:${id}`
  const s = new Set(highlighted.value)
  s.has(key) ? s.delete(key) : s.add(key)
  highlighted.value = s
}

function scrollTo(key) {
  const el = sectionRefs.value[key]
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  const s = new Set(openSections.value)
  s.add(key)
  openSections.value = s
}
</script>

<style scoped>
.ml-tab { display: flex; flex-direction: column; gap: 16px; }

/* ── Loading skeleton ──────────────────────────────────────────────────────── */
.ml-loading { display: flex; flex-direction: column; gap: 12px; }
.ml-sk-banner {
  height: 80px;
  background: var(--card-bg, #1e293b);
  border: 1px solid #334155;
  border-radius: 10px;
  animation: ml-pulse 1.4s ease-in-out infinite;
}
.ml-sk-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
.ml-sk-card {
  height: 100px;
  background: var(--card-bg, #1e293b);
  border: 1px solid #334155;
  border-radius: 10px;
  animation: ml-pulse 1.4s ease-in-out infinite;
}
.ml-sk-section {
  height: 48px;
  background: var(--card-bg, #1e293b);
  border: 1px solid #334155;
  border-radius: 10px;
  animation: ml-pulse 1.4s ease-in-out infinite;
}
@keyframes ml-pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 0.9; } }

/* ── Unavailable ───────────────────────────────────────────────────────────── */
.ml-unavailable { text-align: center; padding: 48px 32px; }
.ml-unavail-icon  { font-size: 36px; margin-bottom: 12px; }
.ml-unavail-title { font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 8px; }
.ml-unavail-body  { font-size: 13px; color: #64748b; max-width: 480px; margin: 0 auto; line-height: 1.6; }

/* ── Gate banner ───────────────────────────────────────────────────────────── */
.ml-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 20px 28px;
  border-left-width: 4px;
  border-left-style: solid;
}
.ml-banner--good  { border-left-color: #4ade80; }
.ml-banner--warn  { border-left-color: #fbbf24; }
.ml-banner--error { border-left-color: #f87171; }

.ml-banner-left { display: flex; flex-direction: column; gap: 8px; }
.ml-gate-badge {
  display: inline-flex;
  align-items: center;
  padding: 5px 16px;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 700;
  width: fit-content;
}
.ml-gate--ready      { background: #0f2718; color: #4ade80; border: 1px solid #166534; }
.ml-gate--needs_work { background: #451a03; color: #fbbf24; border: 1px solid #78350f; }
.ml-gate--not_ready  { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
.ml-gate-desc { font-size: 13px; color: #94a3b8; max-width: 520px; line-height: 1.5; }

.ml-banner-stats { display: flex; gap: 32px; flex-shrink: 0; }
.ml-stat { text-align: center; }
.ml-stat-value { font-size: 24px; font-weight: 800; color: #e2e8f0; }
.ml-stat-label { font-size: 11px; color: #64748b; margin-top: 2px; white-space: nowrap; }

/* ── Summary grid ──────────────────────────────────────────────────────────── */
.ml-summary-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
}
.ml-summary-card {
  background: var(--card-bg, #1e293b);
  border: 1px solid var(--border, #334155);
  border-radius: 10px;
  padding: 16px;
  text-align: left;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.ml-summary-card:hover { background: rgba(255,255,255,0.04); }
.ml-summary-card--red   { border-left: 3px solid #f87171; }
.ml-summary-card--amber { border-left: 3px solid #fbbf24; }
.ml-summary-card--green { border-left: 3px solid #4ade80; }

.ml-sc-header { display: flex; align-items: center; gap: 7px; }
.ml-sc-label  { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; }
.ml-sc-count  { font-size: 20px; font-weight: 700; }
.ml-sc-pct    { font-size: 11px; color: #94a3b8; }
.ml-sc-preview { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.ml-sc-chip {
  font-size: 10px;
  padding: 2px 7px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #94a3b8;
}

/* ── Section ───────────────────────────────────────────────────────────────── */
.ml-section { padding: 0; overflow: visible; }

.ml-section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  cursor: pointer;
  user-select: none;
  transition: background 0.12s;
}
.ml-section-header:hover { background: rgba(255,255,255,0.03); }

.ml-section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  font-weight: 600;
  color: #e2e8f0;
}
.ml-section-count { font-size: 12px; font-weight: 500; }

.ml-section-controls { display: flex; align-items: center; gap: 6px; }
.ml-sort-label { font-size: 11px; color: #64748b; }
.ml-sort-btn {
  padding: 3px 10px;
  font-size: 11px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #64748b;
  cursor: pointer;
  transition: all 0.12s;
  display: flex;
  align-items: center;
  gap: 3px;
}
.ml-sort-btn:hover  { border-color: #60a5fa; color: #93c5fd; }
.ml-sort-btn.active { background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }
.ml-sort-arrow      { font-size: 10px; }

.ml-collapse-btn {
  padding: 3px 8px;
  font-size: 11px;
  background: none;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #64748b;
  cursor: pointer;
  transition: all 0.12s;
  margin-left: 4px;
}
.ml-collapse-btn:hover { border-color: #60a5fa; color: #93c5fd; }

.ml-section-body {
  padding: 0 20px 16px;
  border-top: 1px solid #1e293b;
}
.ml-all-clear { padding: 20px 0; font-size: 13px; color: #4ade80; text-align: center; }

/* ── Findings table ────────────────────────────────────────────────────────── */
.ml-findings-table { width: 100%; margin-top: 12px; }

.ml-findings-header {
  display: grid;
  grid-template-columns: 36px 2fr 1fr 3fr 90px 24px;
  gap: 12px;
  padding: 8px 0;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
  border-bottom: 1px solid #334155;
  position: sticky;
  top: 0;
  background: var(--card-bg, #1e293b);
  z-index: 1;
}

.ml-findings-scroll {
  max-height: 420px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: #334155 transparent;
}
.ml-findings-scroll::-webkit-scrollbar { width: 5px; }
.ml-findings-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

.ml-finding-row {
  display: grid;
  grid-template-columns: 36px 2fr 1fr 3fr 90px 24px;
  gap: 12px;
  align-items: start;
  padding: 9px 0;
  border-bottom: 1px solid #1e293b;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.1s;
}
.ml-finding-row:hover { background: rgba(255,255,255,0.02); }
.ml-finding-row:last-child { border-bottom: none; }
.ml-finding-row--highlighted {
  background: #1e3a5f !important;
  border-left: 2px solid #60a5fa;
  padding-left: 6px;
}

.ml-finding-col {
  font-family: ui-monospace, 'Cascadia Code', monospace;
  font-size: 12px;
  color: #93c5fd;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ml-finding-task  { font-size: 11px; color: #64748b; }
.ml-finding-title { color: #e2e8f0; font-size: 12px; line-height: 1.4; }

/* Expanded detail - spans full grid width */
.ml-finding-detail {
  grid-column: 1 / -1;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 4px;
}
.ml-finding-body {
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.6;
  margin: 0;
}
/* ── Metric stat strip ─────────────────────────────────────────────────────── */
.ml-metric-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 6px;
  overflow: hidden;
}
.ml-metric-stat {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 8px 14px;
  border-right: 1px solid #1e293b;
  flex: 1;
  min-width: 80px;
}
.ml-metric-stat:last-child { border-right: none; }
.ml-metric-stat-label {
  display: block;
  font-size: 10px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  white-space: nowrap;
}
.ml-metric-stat-value {
  display: block;
  font-size: 14px;
  font-weight: 600;
  font-family: ui-monospace, monospace;
  color: #e2e8f0;
}

/* ── Action list ───────────────────────────────────────────────────────────── */
.ml-action-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 4px;
}
.ml-action-list--additional {
  margin-top: 8px;
  padding-top: 12px;
  border-top: 1px solid #1e293b;
}
.ml-action-list-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
  margin-bottom: 2px;
}
.ml-action-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 14px;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 6px;
  transition: border-color 0.12s;
}
.ml-action-row:hover { border-color: #60a5fa; }
.ml-action-row-header {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.ml-action-method {
  font-family: ui-monospace, monospace;
  font-size: 13px;
  font-weight: 600;
  color: #60a5fa;
}
.ml-action-condition {
  font-size: 11px;
  color: #64748b;
  padding: 1px 7px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 4px;
}
.ml-action-what {
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.55;
  margin: 0;
}
.ml-action-tradeoffs,
.ml-action-after {
  display: flex;
  gap: 8px;
  font-size: 12px;
  line-height: 1.5;
}
.ml-tradeoff-label,
.ml-after-label {
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
.ml-tradeoff-text,
.ml-after-text {
  color: #94a3b8;
}
.ml-action-ref {
  font-size: 11px;
  color: #60a5fa;
  text-decoration: none;
  align-self: flex-start;
}
.ml-action-ref:hover { text-decoration: underline; }
.ml-action-cond { color: #64748b; }

.ml-highlight-btn {
  align-self: flex-start;
  padding: 3px 10px;
  font-size: 11px;
  background: none;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #64748b;
  cursor: pointer;
  transition: all 0.12s;
}
.ml-highlight-btn:hover,
.ml-highlight-btn.active { border-color: #60a5fa; color: #60a5fa; background: #1e3a5f; }

.ml-expand-hint {
  font-size: 11px;
  color: #64748b;
  text-align: center;
  transition: color 0.1s;
  align-self: center;
}
.ml-finding-row:hover .ml-expand-hint { color: #94a3b8; }

.ml-mark-btn {
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #64748b;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.12s;
  flex-shrink: 0;
  align-self: center;
}
.ml-mark-btn:hover { border-color: #60a5fa; color: #fbbf24; }
.ml-mark-btn.active { border-color: #fbbf24; color: #fbbf24; background: #451a03; }

/* Strip card shell from embedded TransformationPreviewCard */
.ml-finding-detail :deep(.tx-card) {
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
  box-shadow: none;
}
.ml-finding-detail :deep(.card-title) { display: none; }

/* ── Severity badge ────────────────────────────────────────────────────────── */
.ml-sev-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}
.ml-sev--error { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
.ml-sev--warn  { background: #451a03; color: #fbbf24; border: 1px solid #78350f; }
.ml-sev--info  { background: #0c1a2e; color: #60a5fa; border: 1px solid #1e3a5f; }
.ml-sev--good  { background: #0f2718; color: #4ade80; border: 1px solid #166534; }

/* ── Clean features ────────────────────────────────────────────────────────── */
.ml-clean-scroll {
  max-height: 224px;
  overflow-y: auto;
  padding-top: 12px;
  scrollbar-width: thin;
  scrollbar-color: #334155 transparent;
}
.ml-clean-scroll::-webkit-scrollbar { width: 5px; }
.ml-clean-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
.ml-clean-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.ml-clean-chip {
  font-size: 12px;
  font-family: ui-monospace, 'Cascadia Code', monospace;
  padding: 3px 10px;
  background: #0f2718;
  border: 1px solid #166534;
  border-radius: 4px;
  color: #4ade80;
  cursor: pointer;
  transition: background 0.1s, border-color 0.1s;
}
.ml-clean-chip:hover { background: #14532d; }
.ml-clean-chip--highlighted { background: #1e3a5f; border-color: #60a5fa; color: #93c5fd; }

/* ── Shared colour tokens ──────────────────────────────────────────────────── */
.ml-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  display: inline-block;
}
/* The scorer emits level="green" for passing dimensions; "good" is the
   finding-level badge colour. Both need dot and text colour rules. */
.ml-dot--good,
.ml-dot--green { background: #4ade80; box-shadow: 0 0 5px #4ade8055; }
.ml-dot--warn,
.ml-dot--amber { background: #fbbf24; box-shadow: 0 0 5px #fbbf2455; }
.ml-dot--error,
.ml-dot--red   { background: #f87171; box-shadow: 0 0 5px #f8717155; }

.ml-text--good,
.ml-text--green { color: #4ade80; }
.ml-text--warn,
.ml-text--amber { color: #fbbf24; }
.ml-text--error,
.ml-text--red   { color: #f87171; }

/* ── Expand/collapse transitions ───────────────────────────────────────────── */
.ml-expand-enter-active,
.ml-expand-leave-active { transition: opacity 0.18s, max-height 0.22s ease; max-height: 3000px; overflow: hidden; }
.ml-expand-enter-from,
.ml-expand-leave-to     { opacity: 0; max-height: 0; }

.ml-detail-expand-enter-active,
.ml-detail-expand-leave-active { transition: opacity 0.15s, max-height 0.18s ease; max-height: 600px; overflow: hidden; }
.ml-detail-expand-enter-from,
.ml-detail-expand-leave-to     { opacity: 0; max-height: 0; }

/* ── Light theme ───────────────────────────────────────────────────────────── */
:global(.theme-light) .ml-banner      { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .ml-summary-card { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .ml-section     { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .ml-sc-chip     { background: #f1f5f9; border-color: #cbd5e1; color: #64748b; }
:global(.theme-light) .ml-gate-desc   { color: #64748b; }
:global(.theme-light) .ml-findings-header { background: #f8fafc; border-color: #e2e8f0; color: #94a3b8; }
:global(.theme-light) .ml-finding-row { border-color: #f1f5f9; }
:global(.theme-light) .ml-finding-row:hover { background: rgba(0,0,0,0.02); }
:global(.theme-light) .ml-finding-row--highlighted { background: #eff6ff !important; border-left-color: #2563eb; }
:global(.theme-light) .ml-finding-col { color: #2563eb; }
:global(.theme-light) .ml-finding-title { color: #1e293b; }
:global(.theme-light) .ml-finding-detail { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .ml-finding-body  { color: #64748b; }
:global(.theme-light) .ml-action-chip   { background: #f1f5f9; border-color: #e2e8f0; }
:global(.theme-light) .ml-section-body  { border-top-color: #e2e8f0; }
:global(.theme-light) .ml-section-title { color: #1e293b; }
:global(.theme-light) .ml-clean-chip    { background: #f0fdf4; border-color: #86efac; color: #16a34a; }
:global(.theme-light) .ml-clean-chip--highlighted { background: #eff6ff; border-color: #93c5fd; color: #1d4ed8; }
</style>
