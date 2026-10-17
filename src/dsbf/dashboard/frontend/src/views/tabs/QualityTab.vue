<!-- dsbf/dashboard/frontend/src/views/tabs/QualityTab.vue

  The Quality tab - aggregated data-health findings across all five
  dimensions. Three sections:

    1. Summary cards  - one per dimension, click to jump to that section
    2. Dimension sections - full findings list per dimension, collapsible,
                            sortable by column name / issue / severity
    3. Missingness Mechanism Analysis - separate card below Completeness
    4. Fuzzy Duplicate Detection - separate card below Redundancy
    5. Clean columns  - columns with zero findings across all dimensions

  Data sources
  ────────────
  GET /api/runs/{run_key}/dq-status          → summary cards
  GET /api/runs/{run_key}/tasks/data_quality_scorer → full findings
  tasks prop (pre-loaded by RunDetailView)   → missingness mechanism,
                                               fuzzy duplicates

  Props
  ─────
  runKey : String  (required)
  tasks  : Object  (optional, pre-loaded task results)
-->

<template>
  <div class="quality-tab">

    <!-- ── Loading / error states ─────────────────────────────────────────── -->
    <div v-if="loading" class="qt-loading">
      <div v-for="n in 5" :key="n" class="qt-skeleton-card" />
    </div>

    <div v-else-if="error" class="qt-error">
      {{ error }}
    </div>

    <template v-else-if="!available">
      <div class="qt-unavailable card">
        <div class="qt-unavailable-icon">📊</div>
        <div class="qt-unavailable-title">Data health analysis unavailable</div>
        <div class="qt-unavailable-body">
          Data health findings are not available for this run. Re-run the profiler at standard depth or higher to generate them.
        </div>
      </div>
    </template>

    <template v-else>

      <!-- ── 0. Trust banner ──────────────────────────────────────────────── -->
      <div class="qt-trust-banner card" :class="`qt-trust-banner--${overallLevel}`">
        <div class="qt-trust-left">
          <span class="qt-trust-badge" :class="`qt-trust-badge--${overallLevel}`">
            {{ trustLabel }}
          </span>
          <span class="qt-trust-desc">{{ trustDescription }}</span>
        </div>
        <div class="qt-trust-stats">
          <div class="qt-trust-stat">
            <div class="qt-trust-stat-value">{{ totalColumns }}</div>
            <div class="qt-trust-stat-label">Columns checked</div>
          </div>
          <div class="qt-trust-stat">
            <div class="qt-trust-stat-value qt-text--green">{{ cleanColumns.length }}</div>
            <div class="qt-trust-stat-label">Fully clean</div>
          </div>
          <div class="qt-trust-stat">
            <div class="qt-trust-stat-value" :class="dimensionsWithIssues > 0 ? 'qt-text--amber' : 'qt-text--green'">
              {{ dimensionsWithIssues }} / 5
            </div>
            <div class="qt-trust-stat-label">Dimensions flagged</div>
          </div>
        </div>
      </div>

      <!-- ── 1. Summary cards ─────────────────────────────────────────────── -->
      <div class="qt-summary-grid">
        <button
          v-for="dim in dimensions"
          :key="dim.key"
          class="qt-summary-card"
          :class="`qt-summary-card--${dim.level}`"
          @click="scrollTo(dim.key)"
        >
          <div class="qt-sc-header">
            <span class="qt-sc-dot" :class="`qt-dot--${dim.level}`" />
            <span class="qt-sc-label">{{ dim.label }}</span>
          </div>
          <div class="qt-sc-count" :class="`qt-text--${dim.level}`">
            {{ dim.affectedCount === 0
              ? 'All clear'
              : `${dim.affectedCount} col${dim.affectedCount === 1 ? '' : 's'}` }}
          </div>
          <div class="qt-sc-pct">
            {{ dim.affectedCount === 0
              ? `${totalColumns} columns clean`
              : `${(dim.pctAffected * 100).toFixed(1)}% of dataset` }}
          </div>
          <!-- Issue type breakdown -->
          <div v-if="dim.issueCounts.length" class="qt-sc-preview">
            <span
              v-for="ic in dim.issueCounts"
              :key="ic.issue"
              class="qt-sc-chip"
            >{{ ic.label }}: {{ ic.count }}</span>
          </div>
        </button>
      </div>

      <!-- ── 2. Dimension sections ────────────────────────────────────────── -->
      <template v-for="dim in dimensions" :key="dim.key">
        <div
          :ref="el => sectionRefs[dim.key] = el"
          class="qt-section card"
        >
        <!-- Section header -->
        <div class="qt-section-header" @click="toggleSection(dim.key)">
          <div class="qt-section-title">
            <span class="qt-sc-dot" :class="`qt-dot--${dim.level}`" />
            <span>{{ dim.label }}</span>
            <TooltipIcon
              v-if="DIM_TOOLTIPS[dim.key]"
              :text="DIM_TOOLTIPS[dim.key]"
              direction="down"
              align="left"
            />
            <span class="qt-section-count" :class="`qt-text--${dim.level}`">
              {{ dim.affectedCount === 0
                ? 'All clear'
                : dim.key === 'leakage'
                  ? `${dim.findings.length} pair${dim.findings.length === 1 ? '' : 's'} · ${dim.affectedCount} col${dim.affectedCount === 1 ? '' : 's'}`
                  : dim.findings.length !== dim.affectedCount
                    ? `${dim.affectedCount} col${dim.affectedCount === 1 ? '' : 's'} · ${dim.findings.length} finding${dim.findings.length === 1 ? '' : 's'}`
                    : `${dim.affectedCount} col${dim.affectedCount === 1 ? '' : 's'}` }}
            </span>
          </div>
          <div class="qt-section-controls" @click.stop>
            <!-- Sort controls - only shown when section is open and has findings -->
            <template v-if="openSections.has(dim.key) && dim.affectedCount > 0">
              <span class="qt-sort-label">Sort:</span>
              <button
                v-for="opt in sortOptions(dim.key)"
                :key="opt.key"
                class="qt-sort-btn"
                :class="{ active: sortState[dim.key]?.by === opt.key }"
                @click="setSort(dim.key, opt.key)"
              >
                {{ opt.label }}
                <span v-if="sortState[dim.key]?.by === opt.key" class="qt-sort-arrow">
                  {{ sortState[dim.key].dir === 'asc' ? '↑' : '↓' }}
                </span>
              </button>
            </template>
            <button class="qt-collapse-btn" @click="toggleSection(dim.key)">
              {{ openSections.has(dim.key) ? '▲' : '▼' }}
            </button>
          </div>
        </div>

        <!-- Section body -->
        <Transition name="qt-expand">
          <div v-if="openSections.has(dim.key)" class="qt-section-body">

            <!-- All clear state -->
            <div v-if="dim.affectedCount === 0" class="qt-all-clear">
              ✓ All clear — nothing flagged in this dimension.
            </div>

            <!-- Findings table -->
            <template v-else>
              <div class="qt-findings-table">
                <!-- Header row -->
                <div class="qt-findings-header">
                  <span></span>
                  <span class="qt-col-col">Column</span>
                  <span class="qt-col-issue">Finding</span>
                  <span class="qt-col-detail">Detail</span>
                  <span class="qt-col-sev">Severity</span>
                  <span></span>
                </div>
                <!-- Finding rows - scrollable, max 10 rows visible -->
                <div class="qt-findings-scroll">
                  <div
                    v-for="(finding, i) in sortedFindings(dim)"
                    :key="i"
                    class="qt-finding-row"
                    :class="[
                      `qt-finding-row--${finding.severity}`,
                      { 'qt-finding-row--expanded':    expanded.has(`${dim.key}:${i}`),
                        'qt-finding-row--highlighted': reviewed.has(`${dim.key}:${i}`) }
                    ]"
                    @click="toggleExpanded(dim.key, i)"
                  >
                    <!-- Star / reviewed button -->
                    <button
                      class="qt-mark-btn"
                      :class="{ active: reviewed.has(`${dim.key}:${i}`) }"
                      :title="reviewed.has(`${dim.key}:${i}`) ? 'Marked as reviewed' : 'Mark as reviewed'"
                      @click.stop="toggleReviewed(dim.key, i)"
                    >{{ reviewed.has(`${dim.key}:${i}`) ? '★' : '☆' }}</button>

                    <span class="qt-col-col qt-finding-col">
                      {{ findingColumn(finding) }}
                    </span>
                    <span class="qt-col-issue qt-finding-issue">
                      {{ issueLabel(finding.issue) }}
                    </span>
                    <span class="qt-col-detail qt-finding-detail">
                      {{ findingDetail(finding) }}
                    </span>
                    <span class="qt-col-sev">
                      <span class="qt-sev-badge" :class="`qt-sev--${finding.severity}`">
                        {{ finding.severity }}
                      </span>
                    </span>
                    <span class="qt-expand-hint">
                      {{ expanded.has(`${dim.key}:${i}`) ? '▴' : '▾' }}
                    </span>

                    <!-- Expanded detail - spans full grid width -->
                    <Transition name="qt-detail-expand">
                      <div
                        v-if="expanded.has(`${dim.key}:${i}`)"
                        class="qt-finding-detail-body"
                        @click.stop
                      >
                        <!-- Inline detail: structured metrics + observational note (shown first) -->
                        <div
                          v-if="findingInlineDetail(finding)"
                          class="qt-inline-detail qt-inline-detail--top"
                        >
                          <div
                            v-if="findingInlineDetail(finding).note"
                            class="qt-inline-note"
                          >{{ findingInlineDetail(finding).note }}</div>
                          <div
                            v-if="findingInlineDetail(finding).metrics?.length"
                            class="qt-inline-metrics"
                          >
                            <div
                              v-for="m in findingInlineDetail(finding).metrics"
                              :key="m.label"
                              class="qt-inline-metric"
                            >
                              <span class="qt-inline-metric-label">{{ m.label }}</span>
                              <span class="qt-inline-metric-value">{{ m.value }}</span>
                            </div>
                          </div>
                          <div
                            v-if="findingInlineDetail(finding).samples?.length"
                            class="qt-inline-samples"
                          >
                            <span class="qt-inline-samples-label">Sample values</span>
                            <span
                              v-for="(s, si) in findingInlineDetail(finding).samples"
                              :key="si"
                              class="qt-inline-sample-chip"
                            >{{ s }}</span>
                          </div>
                        </div>

                        <!-- Guidance blurbs from tasks (issue-scoped) -->
                        <template v-if="findingGuidance(finding).length">
                          <div
                            v-for="(blurb, bi) in findingGuidance(finding)"
                            :key="bi"
                            class="qt-blurb"
                            :class="`qt-blurb--${blurb.level}`"
                          >
                            <span class="qt-blurb-icon">{{ blurbIcon(blurb.level) }}</span>
                            <div class="qt-blurb-content">
                              <div class="qt-blurb-title">{{ blurb.title }}</div>
                              <p v-if="blurb.body" class="qt-blurb-body">{{ blurb.body }}</p>
                              <div v-if="blurb.actions?.length" class="qt-blurb-actions">
                                <span
                                  v-for="(act, ai) in blurb.actions"
                                  :key="ai"
                                  class="qt-action-chip"
                                >{{ formatAction(act) }}</span>
                              </div>
                            </div>
                          </div>
                        </template>

                        <!-- Fallback: raw metrics when no guidance -->
                        <div v-else class="qt-blurb-fallback">
                          {{ findingDetail(finding) }} - no additional context available.
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

        <!-- (supplementary cards grouped at bottom) -->

      </template>

      <!-- ── 3. Clean columns ─────────────────────────────────────────────── -->
      <div class="qt-section card">
        <div class="qt-section-header" @click="toggleSection('__clean__')">
          <div class="qt-section-title">
            <span class="qt-sc-dot qt-dot--green" />
            <span>Clean Columns</span>
            <TooltipIcon
              text="Columns with no findings across all five dimensions - no significant missingness, no validity issues, not flagged as IDs or near-constant, not collinear, not part of a leakage pair."
              direction="down"
              align="left"
            />
            <span class="qt-section-count qt-text--green">
              {{ cleanColumns.length }} columns
            </span>
          </div>
          <button class="qt-collapse-btn">
            {{ openSections.has('__clean__') ? '▲' : '▼' }}
          </button>
        </div>

        <Transition name="qt-expand">
          <div v-if="openSections.has('__clean__')" class="qt-section-body">
            <div v-if="cleanColumns.length === 0" class="qt-all-clear">
              All columns have at least one finding.
            </div>
            <div v-else class="qt-clean-scroll">
              <div class="qt-clean-grid">
                <span
                  v-for="col in cleanColumns"
                  :key="col"
                  class="qt-clean-chip"
                  :class="{ 'qt-clean-chip--highlighted': highlighted.has(`__clean__:${col}`) }"
                  @click="toggleHighlight('__clean__', col)"
                >{{ col }}</span>
              </div>
            </div>
          </div>
        </Transition>
      </div>

      <!-- ── 4. Supplementary analysis ──────────────────────────────────────── -->
      <div class="qt-supplementary-label">Supplementary Analysis</div>
      <MissingnessMechanismCard :tasks="tasks" />
      <FuzzyDuplicateCard :tasks="tasks" />

    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { getDqStatus, getTask } from '../../api.js'
import TooltipIcon              from '../../components/TooltipIcon.vue'
import MissingnessMechanismCard from '../../components/quality/MissingnessMechanismCard.vue'
import FuzzyDuplicateCard       from '../../components/quality/FuzzyDuplicateCard.vue'

const props = defineProps({
  runKey: { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
})

// ── State ─────────────────────────────────────────────────────────────────────

const loading      = ref(true)
const error        = ref(null)
const available    = ref(false)
const totalColumns = ref(0)
const allColumns   = ref([])   // full ordered column list from scorer
const categories   = ref({})

// Section collapse - start all collapsed; open sections with findings once data loads
const openSections = ref(new Set())

// Sort state per dimension: { by: 'severity'|'column'|'issue', dir: 'asc'|'desc' }
const sortState = ref({
  completeness: { by: 'severity', dir: 'desc' },
  validity:     { by: 'severity', dir: 'desc' },
  usability:    { by: 'severity', dir: 'desc' },
  redundancy:   { by: 'severity', dir: 'desc' },
  leakage:      { by: 'severity', dir: 'desc' },
})

// Section element refs for scroll-to
const sectionRefs = ref({})

// expanded: Set of "dimKey:rowIndex" - which finding rows are open
const expanded  = ref(new Set())
// reviewed: Set of "dimKey:rowIndex" - which findings are starred/marked
const reviewed  = ref(new Set())
// highlighted: kept for clean-column chip highlighting
const highlighted = ref(new Set())

function toggleExpanded(dimKey, i) {
  const key = `${dimKey}:${i}`
  const s = new Set(expanded.value)
  s.has(key) ? s.delete(key) : s.add(key)
  expanded.value = s
}

function toggleReviewed(dimKey, i) {
  const key = `${dimKey}:${i}`
  const s = new Set(reviewed.value)
  s.has(key) ? s.delete(key) : s.add(key)
  reviewed.value = s
}

function toggleHighlight(dimKey, id) {
  const key = `${dimKey}:${id}`
  const s = new Set(highlighted.value)
  s.has(key) ? s.delete(key) : s.add(key)
  highlighted.value = s
}

// ── Fetch ─────────────────────────────────────────────────────────────────────

async function fetchData(runKey) {
  loading.value    = true
  error.value      = null
  available.value  = false
  categories.value = {}

  try {
    // Step 1: check availability via the lightweight dq-status endpoint.
    // This never 404s for a valid run - it returns { available: false } when
    // the scorer was skipped or not yet run (e.g. a failed dependency caused
    // the scorer to be skipped by the graph executor).
    const status = await getDqStatus(runKey)
    available.value    = status.available ?? false
    totalColumns.value = status.total_columns ?? 0

    if (!available.value) return

    // Step 2: only fetch the full task result when the scorer actually ran.
    // Calling getTask when the scorer was skipped would 404 - skipped tasks
    // are never written to the database.
    const task = await getTask(runKey, 'data_quality_scorer')
    if (task?.data?.categories) {
      categories.value = task.data.categories
    }
    if (task?.data?.all_columns) {
      allColumns.value = task.data.all_columns
    }
  } catch (e) {
    error.value = `Failed to load quality data: ${e.message}`
  } finally {
    loading.value = false
  }
}

onMounted(() => fetchData(props.runKey))
watch(() => props.runKey, key => { if (key) fetchData(key) })

// Auto-open sections that have findings once data arrives.
// Watch categories (defined above) rather than dimensions (defined below)
// to avoid a temporal dead zone error during setup.
watch(categories, (cats) => {
  const s = new Set(openSections.value)
  for (const [key, cat] of Object.entries(cats)) {
    if ((cat.affected_count ?? 0) > 0) s.add(key)
  }
  openSections.value = s
}, { immediate: false })

// ── Dimension metadata ────────────────────────────────────────────────────────

const DIMENSION_META = {
  completeness: { label: 'Completeness', order: 0 },
  validity:     { label: 'Validity',     order: 1 },
  usability:    { label: 'Usability',    order: 2 },
  redundancy:   { label: 'Redundancy',   order: 3 },
  leakage:      { label: 'Leakage',      order: 4 },
}

const SEVERITY_RANK = { error: 3, warn: 2, info: 1 }

const dimensions = computed(() =>
  Object.entries(DIMENSION_META)
    .sort((a, b) => a[1].order - b[1].order)
    .map(([key, meta]) => {
      const cat      = categories.value[key] ?? {}
      const findings = cat.findings ?? []
      // Count occurrences of each issue type for the summary card
      const issueCountMap = {}
      for (const f of findings) {
        const lbl = ISSUE_LABELS[f.issue] ?? f.issue?.replace(/_/g, ' ') ?? 'other'
        issueCountMap[lbl] = (issueCountMap[lbl] ?? 0) + 1
      }
      const issueCounts = Object.entries(issueCountMap)
        .sort((a, b) => b[1] - a[1])
        .map(([label, count]) => ({ label, count }))

      return {
        key,
        label:         meta.label,
        level:         cat.level         ?? 'green',
        affectedCount: cat.affected_count ?? 0,
        pctAffected:   cat.pct_affected   ?? 0,
        findings,
        issueCounts,
      }
    })
)

// ── Clean columns ─────────────────────────────────────────────────────────────

const cleanColumns = computed(() => {
  if (!available.value || !allColumns.value.length) return []

  // Use the authoritative full column list from the scorer output.
  // Deriving it from findings misses columns that have zero findings
  // across all dimensions - they never appear in any finding and would
  // be incorrectly omitted from the clean list.
  const affected = new Set()
  for (const cat of Object.values(categories.value)) {
    for (const col of cat.affected_columns ?? []) {
      affected.add(col)
    }
  }

  return allColumns.value.filter(col => !affected.has(col)).sort()
})

// ── Trust banner ──────────────────────────────────────────────────────────────

const LEVEL_RANK = { green: 0, blue: 1, amber: 2, red: 3 }

const overallLevel = computed(() => {
  const levels = dimensions.value.map(d => d.level)
  if (levels.includes('red'))   return 'red'
  if (levels.includes('amber')) return 'amber'
  if (levels.includes('blue'))  return 'blue'
  return 'green'
})

const dimensionsWithIssues = computed(() =>
  dimensions.value.filter(d => d.affectedCount > 0).length
)

const trustLabel = computed(() => ({
  green: '✓  Looking Good',
  blue:  'ℹ  A Few Notes',
  amber: '⚠  A Few Things to Note',
  red:   '⚠  Worth Investigating',
}[overallLevel.value]))

const trustDescription = computed(() => ({
  green: 'No notable data quality issues found across all five dimensions. Good to explore.',
  blue:  'A small number of informational observations. Nothing requires action — expand any blue section to read the notes.',
  amber: 'Some columns flagged across one or more dimensions. Worth reviewing before drawing conclusions.',
  red:   'Several quality signals detected. Keep these in mind as you explore - they may affect how you interpret results.',
}[overallLevel.value]))

// ── Guidance helpers ──────────────────────────────────────────────────────────

/** Collect all EDA-phase guidance blurbs for the column in a finding */
// Dimension tooltips - describe data sources, not the general concept (the
// health bar already covers the general definition).
const DIM_TOOLTIPS = {
  completeness: 'Findings from null analysis. Flags columns where a significant proportion of values are missing.',
  validity:     'Findings from constant column detection, out-of-bounds value checks, and structural zero analysis.',
  usability:    'Findings from ID column detection, dominant value analysis, and high cardinality checks - columns that may not behave as grouping variables.',
  redundancy:   'Findings from collinearity detection (VIF). Flags columns whose variance is largely explained by other columns in the dataset.',
  leakage:      'Findings from near-perfect correlation detection and exact duplicate column checks. Flags column pairs that appear to encode the same information.',
}

// Maps each finding issue type to the task(s) whose guidance is relevant to it.
// This prevents blurbs from unrelated tasks (bimodal, kurtosis, dtype, etc.)
// from appearing in expanded rows where they would be confusing and off-topic.
const ISSUE_TO_TASKS = {
  missing_values:   ['summarize_nulls', 'missingness_mechanism_analysis'],
  out_of_bounds:    ['detect_out_of_bounds'],
  constant_column:  ['detect_constant_columns'],
  structural_zeros: ['detect_zeros'],
  likely_id:        ['detect_id_columns'],
  dominant_value:   ['detect_single_dominant_value'],
  high_cardinality: ['detect_high_cardinality'],
  high_vif:         ['detect_collinear_features'],
  leakage_pair:     ['detect_data_leakage', 'detect_duplicate_columns'],
}

const BLURB_LEVEL_RANK = { error: 0, warn: 1, info: 2, good: 3 }

function findingGuidance(finding) {
  const col        = finding.column ?? finding.col_a ?? null
  if (!col) return []
  const allowedTasks = new Set(ISSUE_TO_TASKS[finding.issue] ?? [])
  const out = []
  for (const [taskName, task] of Object.entries(props.tasks)) {
    if (allowedTasks.size && !allowedTasks.has(taskName)) continue
    const blurbs = task?.guidance?.[col]?.eda
    if (Array.isArray(blurbs)) out.push(...blurbs)
  }
  return out.sort((a, b) => {
    const levelDiff = (BLURB_LEVEL_RANK[a.level] ?? 99) - (BLURB_LEVEL_RANK[b.level] ?? 99)
    if (levelDiff !== 0) return levelDiff
    return (a.title ?? '').localeCompare(b.title ?? '')
  })
}

function blurbIcon(level) {
  return { error: '🚫', warn: '⚠️', info: 'ℹ️', good: '✅' }[level] ?? 'ℹ️'
}

function formatAction(action) {
  if (typeof action === 'string') return action
  if (action.method) {
    return action.condition || action.detail
      ? `${action.method} - ${action.condition ?? action.detail}`
      : action.method
  }
  return action.action ?? ''
}

/** Extract the primary column name from a finding (handles pair findings) */
function findingColumn(finding) {
  if (finding.column) return finding.column
  if (finding.col_a)  return `${finding.col_a} / ${finding.col_b}`
  return '-'
}

const ISSUE_LABELS = {
  missing_values:    'Missing values',
  out_of_bounds:     'Out of bounds',
  constant_column:   'Constant column',
  structural_zeros:  'Structural zeros',
  likely_id:         'Likely ID column',
  dominant_value:    'Dominant value',
  high_cardinality:  'High cardinality',
  high_vif:          'Multicollinearity',
  leakage_pair:      'Leakage pair',
}

function issueLabel(issue) {
  return ISSUE_LABELS[issue] ?? issue?.replace(/_/g, ' ') ?? '-'
}

/** Build a human-readable detail string from a finding's metrics */
function findingDetail(finding) {
  if (finding.pct_null     != null) return `${(finding.pct_null * 100).toFixed(1)}% missing`
  if (finding.violation_count != null) return `${finding.violation_count} violation${finding.violation_count === 1 ? '' : 's'}`
  if (finding.pct_zero     != null) return `${(finding.pct_zero * 100).toFixed(1)}% zeros`
  if (finding.mode_proportion != null) return `${(finding.mode_proportion * 100).toFixed(1)}% single value`
  if (finding.n_unique     != null) return `${finding.n_unique.toLocaleString()} unique values`
  if (finding.vif_score    != null) return `VIF ${finding.vif_score.toFixed(1)}`
  if (finding.correlation  != null) return `r = ${finding.correlation.toFixed(3)}`
  return '-'
}

/**
 * Returns structured extra detail for the expanded finding body.
 * Completes the "where and what" for each issue type without making
 * preparation recommendations - purely observational.
 */
function findingInlineDetail(finding) {
  const numRows = props.tasks?.summarize_dataset_shape?.data?.num_rows ?? null

  // ── Completeness: missing values ─────────────────────────────────────────
  if (finding.issue === 'missing_values' && finding.pct_null != null) {
    const absCount = numRows != null
      ? Math.round(finding.pct_null * numRows).toLocaleString()
      : null
    return {
      note: absCount
        ? `${absCount} of ${numRows.toLocaleString()} rows are missing this value (${(finding.pct_null * 100).toFixed(1)}%).`
        : null,
      metrics: null,
    }
  }

  // ── Validity: constant column ─────────────────────────────────────────────
  if (finding.issue === 'constant_column') {
    const col = finding.column
    // Try to get constant value from value_counts (first key) or numeric stats
    const vc  = props.tasks?.summarize_value_counts?.data?.[col]
    const nm  = props.tasks?.summarize_numeric?.data?.[col]
    const val = vc ? Object.keys(vc)[0]
              : nm?.min != null ? String(nm.min)
              : null
    return {
      note: val != null
        ? `Every row contains the same value: "${val}". This column carries no information and cannot distinguish between observations.`
        : 'Every row contains the same value. This column carries no information and cannot distinguish between observations.',
      metrics: null,
    }
  }

  // ── Validity: structural zeros ────────────────────────────────────────────
  if (finding.issue === 'structural_zeros' && finding.pct_zero != null) {
    const col = finding.column
    const absCount = numRows != null
      ? Math.round(finding.pct_zero * numRows).toLocaleString()
      : null
    return {
      note: absCount
        ? `${absCount} of ${numRows.toLocaleString()} rows (${(finding.pct_zero * 100).toFixed(1)}%) are zero. A very high proportion of zeros often indicates a structural empty rather than a measured value.`
        : `${(finding.pct_zero * 100).toFixed(1)}% of rows are zero.`,
      metrics: null,
    }
  }

  // ── Validity: out of bounds ───────────────────────────────────────────────
  if (finding.issue === 'out_of_bounds') {
    const col = finding.column
    const oob = props.tasks?.detect_out_of_bounds?.data?.[col]
    const metrics = []
    if (oob?.violation_count != null)
      metrics.push({ label: 'Violations', value: oob.violation_count.toLocaleString() })
    if (oob?.min_violation != null)
      metrics.push({ label: 'Min violation', value: String(oob.min_violation) })
    if (oob?.max_violation != null)
      metrics.push({ label: 'Max violation', value: String(oob.max_violation) })
    return {
      note: 'Values were found outside the expected domain for this column.',
      metrics: metrics.length ? metrics : null,
    }
  }

  // ── Usability: likely ID ──────────────────────────────────────────────────
  if (finding.issue === 'likely_id') {
    const col = finding.column
    const vc  = props.tasks?.summarize_value_counts?.data?.[col] ?? {}
    const samples = Object.keys(vc).slice(0, 3)
    const unique  = props.tasks?.summarize_unique?.data?.[col]
    return {
      note: `This column has ${unique != null ? unique.toLocaleString() + ' unique values - ' : ''}nearly every row is distinct, which is characteristic of an identifier rather than a grouping variable.`,
      samples: samples.length ? samples : null,
      metrics: null,
    }
  }

  // ── Usability: dominant value ─────────────────────────────────────────────
  if (finding.issue === 'dominant_value') {
    const col  = finding.column
    const dom  = props.tasks?.detect_single_dominant_value?.data?.[col]
    const mode = dom?.mode != null ? String(dom.mode) : null
    const pct  = finding.mode_proportion != null
      ? (finding.mode_proportion * 100).toFixed(1) + '%'
      : null
    return {
      note: mode && pct
        ? `"${mode}" appears in ${pct} of rows. Columns dominated by a single value provide limited ability to distinguish between observations.`
        : pct
          ? `A single value appears in ${pct} of rows.`
          : null,
      metrics: null,
    }
  }

  // ── Usability: high cardinality ───────────────────────────────────────────
  if (finding.issue === 'high_cardinality') {
    const col   = finding.column
    const n     = finding.n_unique
    const ratio = numRows && n ? ((n / numRows) * 100).toFixed(1) + '%' : null
    return {
      note: ratio
        ? `${n?.toLocaleString()} unique values - ${ratio} of all rows are distinct. When almost every value is unique, this column behaves more like an identifier than a grouping variable.`
        : `${n?.toLocaleString()} unique values.`,
      metrics: null,
    }
  }

  // ── Redundancy: high VIF ──────────────────────────────────────────────────
  if (finding.issue === 'high_vif' && finding.vif_score != null) {
    const col = finding.column
    const vif = finding.vif_score
    const bracket = vif > 100 ? 'extremely high'
                  : vif > 30  ? 'very high'
                  : vif > 10  ? 'high'
                  : 'elevated'
    return {
      note: `VIF of ${vif.toFixed(1)} indicates ${bracket} collinearity - this column's variance is largely explained by other columns in the dataset. It is not independent information.`,
      metrics: null,
    }
  }

  // ── Leakage: leakage pair ────────────────────────────────────────────────
  if (finding.issue === 'leakage_pair' && finding.correlation != null) {
    const r    = Math.abs(finding.correlation)
    const desc = r >= 1.0
      ? 'These two columns are perfectly correlated - knowing one tells you exactly what the other is.'
      : `These two columns move together almost perfectly (r = ${r.toFixed(3)}). They appear to encode the same or nearly the same information.`
    return {
      note: desc,
      metrics: [
        { label: 'Column A',     value: finding.col_a },
        { label: 'Column B',     value: finding.col_b },
        { label: 'Correlation',  value: `r = ${finding.correlation.toFixed(4)}` },
      ],
      samples: null,
    }
  }

  return null
}

// ── Sort options per dimension ─────────────────────────────────────────────────

function sortOptions(dimKey) {
  const base = [
    { key: 'column',   label: 'Column'   },
    { key: 'issue',    label: 'Finding'  },
    { key: 'severity', label: 'Severity' },
  ]
  // Leakage findings are pairs - no single column to sort on
  if (dimKey === 'leakage') {
    return base.filter(o => o.key !== 'column')
  }
  return base
}

function setSort(dimKey, by) {
  const cur = sortState.value[dimKey]
  if (cur.by === by) {
    // Toggle direction
    sortState.value = {
      ...sortState.value,
      [dimKey]: { by, dir: cur.dir === 'asc' ? 'desc' : 'asc' },
    }
  } else {
    // New field - default direction: severity desc, others asc
    sortState.value = {
      ...sortState.value,
      [dimKey]: { by, dir: by === 'severity' ? 'desc' : 'asc' },
    }
  }
}

function sortedFindings(dim) {
  const { by, dir } = sortState.value[dim.key] ?? { by: 'severity', dir: 'desc' }
  const mult = dir === 'asc' ? 1 : -1

  return [...dim.findings].sort((a, b) => {
    if (by === 'severity') {
      return mult * ((SEVERITY_RANK[a.severity] ?? 0) - (SEVERITY_RANK[b.severity] ?? 0))
    }
    if (by === 'column') {
      const ca = findingColumn(a) ?? ''
      const cb = findingColumn(b) ?? ''
      return mult * ca.localeCompare(cb)
    }
    if (by === 'issue') {
      return mult * (issueLabel(a.issue) ?? '').localeCompare(issueLabel(b.issue) ?? '')
    }
    return 0
  })
}

// ── Collapse / expand ─────────────────────────────────────────────────────────

function toggleSection(key) {
  const s = new Set(openSections.value)
  s.has(key) ? s.delete(key) : s.add(key)
  openSections.value = s
}

// ── Scroll to section ─────────────────────────────────────────────────────────

function scrollTo(key) {
  const el = sectionRefs.value[key]
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  // Ensure section is open
  const s = new Set(openSections.value)
  s.add(key)
  openSections.value = s
}
</script>

<style scoped>
.quality-tab {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* ── Trust banner ──────────────────────────────────────────────────────────── */
.qt-trust-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 20px 24px;
  border-left: 4px solid;
  flex-wrap: wrap;
}
.qt-trust-banner--green { border-left-color: #4ade80; }
.qt-trust-banner--blue  { border-left-color: #60a5fa; }
.qt-trust-banner--amber { border-left-color: #fbbf24; }
.qt-trust-banner--red   { border-left-color: #f87171; }

.qt-trust-left {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  flex: 1;
}

.qt-trust-badge {
  font-size: 13px;
  font-weight: 700;
  padding: 4px 14px;
  border-radius: 6px;
  border: 1px solid;
  white-space: nowrap;
}
.qt-trust-badge--green { background: #0f2718; color: #4ade80; border-color: #4ade80; }
.qt-trust-badge--blue  { background: #0c1a2e; color: #60a5fa; border-color: #60a5fa; }
.qt-trust-badge--amber { background: #3d2a00; color: #fbbf24; border-color: #fbbf24; }
.qt-trust-badge--red   { background: #3d0f0f; color: #f87171; border-color: #f87171; }

.qt-trust-desc {
  font-size: 13px;
  color: #94a3b8;
  line-height: 1.4;
}

.qt-trust-stats {
  display: flex;
  gap: 32px;
  flex-shrink: 0;
}

.qt-trust-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}
.qt-trust-stat-value {
  font-size: 22px;
  font-weight: 700;
  color: #f1f5f9;
  line-height: 1;
}
.qt-trust-stat-label {
  font-size: 10px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  white-space: nowrap;
}

/* ── Supplementary section label ───────────────────────────────────────────── */
.qt-supplementary-label {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: #334155;
  padding: 4px 2px;
  margin-top: 4px;
}

/* ── Finding row grid (updated for star + expand hint columns) ─────────────── */
.qt-findings-header,
.qt-finding-row {
  display: grid;
  grid-template-columns: 32px 2fr 2fr 2fr 100px 20px;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
}

/* ── Star / reviewed button ────────────────────────────────────────────────── */
.qt-mark-btn {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #475569;
  cursor: pointer;
  font-size: 12px;
  padding: 0;
  transition: all 0.12s;
  flex-shrink: 0;
}
.qt-mark-btn:hover { border-color: #fbbf24; color: #fbbf24; }
.qt-mark-btn.active { border-color: #fbbf24; color: #fbbf24; background: #451a03; }

/* ── Expand hint ───────────────────────────────────────────────────────────── */
.qt-expand-hint {
  font-size: 13px;
  color: #475569;
  text-align: center;
  transition: color 0.1s;
}
.qt-finding-row:hover .qt-expand-hint { color: #94a3b8; }

/* ── Expanded finding detail ───────────────────────────────────────────────── */
.qt-finding-detail-body {
  grid-column: 1 / -1;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 6px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 4px;
}

.qt-blurb {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 6px;
  border-left: 3px solid;
}
.qt-blurb--error { background: #3d0f0f; border-color: #f87171; }
.qt-blurb--warn  { background: #3d2510; border-color: #fb923c; }
.qt-blurb--info  { background: #1e3a5f; border-color: #60a5fa; }
.qt-blurb--good  { background: #0f2718; border-color: #4ade80; }

.qt-blurb-icon { font-size: 13px; flex-shrink: 0; margin-top: 1px; }

.qt-blurb-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
}

.qt-blurb-title {
  font-size: 12px;
  font-weight: 600;
}
.qt-blurb--error .qt-blurb-title { color: #fca5a5; }
.qt-blurb--warn  .qt-blurb-title { color: #fed7aa; }
.qt-blurb--info  .qt-blurb-title { color: #bfdbfe; }
.qt-blurb--good  .qt-blurb-title { color: #bbf7d0; }

.qt-blurb-body {
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.55;
  margin: 0;
}

.qt-blurb-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 2px;
}

.qt-action-chip {
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 10px;
  background: #0f172a;
  color: #64748b;
  border: 1px solid #334155;
  white-space: nowrap;
}

.qt-blurb-fallback {
  font-size: 12px;
  color: #64748b;
  font-style: italic;
}

/* ── Inline finding detail ───────────────────────────────────────────────── */
.qt-inline-detail {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* When at the top, add bottom border to separate from blurbs below */
.qt-inline-detail--top {
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid #1e293b;
}
.qt-inline-note {
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.6;
}
.qt-inline-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
}
.qt-inline-metric {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.qt-inline-metric-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #64748b;
  font-weight: 600;
}
.qt-inline-metric-value {
  font-size: 13px;
  font-weight: 600;
  font-family: ui-monospace, monospace;
  color: #e2e8f0;
}
.qt-inline-samples {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.qt-inline-samples-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #64748b;
  font-weight: 600;
  flex-shrink: 0;
}
.qt-inline-sample-chip {
  font-size: 11px;
  font-family: ui-monospace, monospace;
  color: #94a3b8;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 4px;
  padding: 2px 8px;
}
.qt-inline-link {
  font-size: 12px;
  color: #60a5fa;
  text-decoration: none;
  align-self: flex-start;
  cursor: pointer;
}
.qt-inline-link:hover { text-decoration: underline; }

/* ── Detail expand transition ──────────────────────────────────────────────── */
.qt-detail-expand-enter-active,
.qt-detail-expand-leave-active {
  transition: opacity 0.15s, max-height 0.18s ease;
  max-height: 600px;
  overflow: hidden;
}
.qt-detail-expand-enter-from,
.qt-detail-expand-leave-to { opacity: 0; max-height: 0; }

/* ── Light theme additions ─────────────────────────────────────────────────── */
:global(.theme-light) .qt-trust-badge--green { background: #f0fdf4; }
:global(.theme-light) .qt-trust-badge--blue  { background: #eff6ff; }
:global(.theme-light) .qt-trust-badge--amber { background: #fffbeb; }
:global(.theme-light) .qt-trust-badge--red   { background: #fef2f2; }
:global(.theme-light) .qt-trust-desc         { color: #64748b; }
:global(.theme-light) .qt-trust-stat-value   { color: #1e293b; }
:global(.theme-light) .qt-supplementary-label { color: #94a3b8; }
:global(.theme-light) .qt-finding-detail-body { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .qt-blurb-body         { color: #64748b; }
:global(.theme-light) .qt-blurb-fallback     { color: #94a3b8; }
:global(.theme-light) .qt-action-chip        { background: #f1f5f9; border-color: #e2e8f0; }
:global(.theme-light) .qt-inline-detail      { border-bottom-color: #e2e8f0; }
:global(.theme-light) .qt-inline-note        { color: #64748b; }
:global(.theme-light) .qt-inline-metric-value { color: #1e293b; }
:global(.theme-light) .qt-inline-sample-chip { background: #f8fafc; border-color: #e2e8f0; color: #64748b; }


.qt-loading {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
}
.qt-skeleton-card {
  height: 110px;
  background: var(--card-bg, #1e293b);
  border: 1px solid var(--border, #334155);
  border-radius: 10px;
  animation: qt-pulse 1.4s ease-in-out infinite;
}
@keyframes qt-pulse {
  0%, 100% { opacity: 0.5; }
  50%       { opacity: 1; }
}

/* ── Unavailable state ─────────────────────────────────────────────────────── */
.qt-unavailable {
  text-align: center;
  padding: 48px 32px;
}
.qt-unavailable-icon  { font-size: 36px; margin-bottom: 12px; }
.qt-unavailable-title { font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 8px; }
.qt-unavailable-body  { font-size: 13px; color: #64748b; max-width: 480px; margin: 0 auto; line-height: 1.6; }

/* ── Summary grid ──────────────────────────────────────────────────────────── */
.qt-summary-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
}

.qt-summary-card {
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
.qt-summary-card:hover { background: var(--hover-bg, rgba(255,255,255,0.04)); }
.qt-summary-card--red   { border-left: 3px solid #f87171; }
.qt-summary-card--amber { border-left: 3px solid #fbbf24; }
.qt-summary-card--blue  { border-left: 3px solid #60a5fa; }
.qt-summary-card--green { border-left: 3px solid #4ade80; }

.qt-sc-header {
  display: flex;
  align-items: center;
  gap: 7px;
}
.qt-sc-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
}
.qt-sc-count {
  font-size: 20px;
  font-weight: 700;
}
.qt-sc-pct {
  font-size: 11px;
  color: #475569;
}
.qt-sc-preview {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 4px;
}
.qt-sc-chip {
  font-size: 10px;
  padding: 2px 7px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #94a3b8;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 120px;
}
.qt-sc-chip--more {
  color: #475569;
  border-style: dashed;
}

/* ── Traffic light dots ────────────────────────────────────────────────────── */
.qt-sc-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  display: inline-block;
}
.qt-dot--green { background: #4ade80; box-shadow: 0 0 5px #4ade8055; }
.qt-dot--blue  { background: #60a5fa; box-shadow: 0 0 5px #60a5fa55; }
.qt-dot--amber { background: #fbbf24; box-shadow: 0 0 5px #fbbf2455; }
.qt-dot--red   { background: #f87171; box-shadow: 0 0 5px #f8717155; }

/* ── Colour text variants ──────────────────────────────────────────────────── */
.qt-text--green { color: #4ade80; }
.qt-text--blue  { color: #60a5fa; }
.qt-text--amber { color: #fbbf24; }
.qt-text--red   { color: #f87171; }

/* ── Section card ──────────────────────────────────────────────────────────── */
.qt-section { padding: 0; overflow: visible; }

.qt-section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  cursor: pointer;
  user-select: none;
  transition: background 0.12s;
}
.qt-section-header:hover { background: rgba(255,255,255,0.03); }

.qt-section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  font-weight: 600;
  color: #e2e8f0;
}
.qt-section-count {
  font-size: 12px;
  font-weight: 500;
}

.qt-section-controls {
  display: flex;
  align-items: center;
  gap: 6px;
}
.qt-sort-label {
  font-size: 11px;
  color: #475569;
}
.qt-sort-btn {
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
.qt-sort-btn:hover  { border-color: #60a5fa; color: #93c5fd; }
.qt-sort-btn.active { background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }
.qt-sort-arrow      { font-size: 10px; }

.qt-collapse-btn {
  padding: 3px 8px;
  font-size: 11px;
  background: none;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #475569;
  cursor: pointer;
  transition: all 0.12s;
  margin-left: 4px;
}
.qt-collapse-btn:hover { border-color: #60a5fa; color: #93c5fd; }

/* ── Section body ──────────────────────────────────────────────────────────── */
.qt-section-body {
  padding: 0 20px 16px;
  border-top: 1px solid #1e293b;
}

.qt-all-clear {
  padding: 20px 0;
  font-size: 13px;
  color: #4ade80;
  text-align: center;
}

/* ── Findings table ────────────────────────────────────────────────────────── */
.qt-findings-table {
  width: 100%;
  margin-top: 12px;
}

/* Scrollable container: shows ~10 rows before scrolling */
.qt-findings-scroll {
  max-height: 420px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: #334155 transparent;
}
.qt-findings-scroll::-webkit-scrollbar { width: 5px; }
.qt-findings-scroll::-webkit-scrollbar-track { background: transparent; }
.qt-findings-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

.qt-findings-header {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #475569;
  border-bottom: 1px solid #334155;
  padding-bottom: 8px;
  position: sticky;
  top: 0;
  background: var(--card-bg, #1e293b);
  z-index: 1;
}

.qt-finding-row {
  border-bottom: 1px solid #1e293b;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.1s;
}
.qt-finding-row:hover { background: rgba(255,255,255,0.03); }
.qt-finding-row:last-child { border-bottom: none; }
.qt-finding-row--highlighted {
  background: #1e3a5f !important;
  border-left: 2px solid #60a5fa;
  padding-left: 6px;
}

.qt-finding-col {
  font-family: ui-monospace, 'Cascadia Code', monospace;
  font-size: 12px;
  color: #93c5fd;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.qt-finding-issue  { color: #e2e8f0; }
.qt-finding-detail { color: #94a3b8; font-size: 12px; }

/* ── Severity badge ────────────────────────────────────────────────────────── */
.qt-sev-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}
.qt-sev--error { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
.qt-sev--warn  { background: #451a03; color: #fbbf24; border: 1px solid #78350f; }
.qt-sev--info  { background: #0c1a2e; color: #60a5fa; border: 1px solid #1e3a5f; }

/* ── Clean columns ─────────────────────────────────────────────────────────── */
.qt-clean-scroll {
  max-height: 224px; /* ~4 rows of chips before scrolling */
  overflow-y: auto;
  padding-top: 12px;
  scrollbar-width: thin;
  scrollbar-color: #334155 transparent;
}
.qt-clean-scroll::-webkit-scrollbar { width: 5px; }
.qt-clean-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

.qt-clean-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.qt-clean-chip {
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
.qt-clean-chip:hover { background: #14532d; }
.qt-clean-chip--highlighted {
  background: #1e3a5f;
  border-color: #60a5fa;
  color: #93c5fd;
}

/* ── Expand/collapse transition ────────────────────────────────────────────── */
.qt-expand-enter-active,
.qt-expand-leave-active {
  transition: opacity 0.18s, max-height 0.22s ease;
  max-height: 2000px;
  overflow: hidden;
}
.qt-expand-enter-from,
.qt-expand-leave-to {
  opacity: 0;
  max-height: 0;
}

/* ── Light theme overrides ─────────────────────────────────────────────────── */
:global(.theme-light) .qt-summary-card,
:global(.theme-light) .qt-section       { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .qt-sc-chip       { background: #f1f5f9; border-color: #cbd5e1; color: #64748b; }
:global(.theme-light) .qt-sort-btn,
:global(.theme-light) .qt-collapse-btn  { background: #f1f5f9; border-color: #cbd5e1; }
:global(.theme-light) .qt-findings-header { color: #94a3b8; border-color: #e2e8f0; background: #f8fafc; }
:global(.theme-light) .qt-finding-row   { border-color: #f1f5f9; }
:global(.theme-light) .qt-finding-row:hover { background: rgba(0,0,0,0.02); }
:global(.theme-light) .qt-finding-row--highlighted { background: #eff6ff !important; border-left-color: #2563eb; }
:global(.theme-light) .qt-finding-col   { color: #2563eb; }
:global(.theme-light) .qt-finding-issue { color: #1e293b; }
:global(.theme-light) .qt-clean-chip    { background: #f0fdf4; border-color: #86efac; color: #16a34a; }
:global(.theme-light) .qt-clean-chip--highlighted { background: #eff6ff; border-color: #93c5fd; color: #1d4ed8; }
:global(.theme-light) .qt-section-body  { border-top-color: #e2e8f0; }
:global(.theme-light) .qt-sc-label      { color: #94a3b8; }
:global(.theme-light) .qt-sc-pct        { color: #94a3b8; }
:global(.theme-light) .qt-section-title { color: #1e293b; }
:global(.theme-light) .qt-unavailable-title { color: #1e293b; }
</style>
