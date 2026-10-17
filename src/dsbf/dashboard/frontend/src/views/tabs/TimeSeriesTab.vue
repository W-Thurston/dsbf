<!-- dsbf/dashboard/frontend/src/views/tabs/TimeSeriesTab.vue

  Time Series tab. Always visible in the tab bar.

  States:
  - disabled : time_series.enabled is false in config (default) - shows a
               friendly empty state with exact config instructions
  - no_index : enabled but datetime_index_column not set or not found
  - ready    : at least one TS task produced results

  Three sections when ready:
    1. ACF / PACF        - significant lag counts per value column
    2. Stationarity      - ADF/KPSS assessment per value column
    3. STL Decomposition - trend/seasonal strength per value column

  All sections are data-only summaries for now (no D3 charts yet).
  Each shows the raw numerical results in a readable table format.
  Charts will be added during the D3/Vue refactor.

  Props
  ─────
  tasks : Object  - pre-loaded task results from RunDetailView
  theme : String  - "dark" | "light"
-->

<template>
  <div class="ts-tab">

    <!-- ── Disabled empty state ──────────────────────────────────────────── -->
    <div v-if="tsState === 'disabled'" class="card ts-disabled">
      <div class="ts-disabled-icon">📈</div>
      <div class="ts-disabled-title">Time Series Analysis Not Enabled</div>
      <div class="ts-disabled-body">
        DSBF does not guess which datetime column represents the time axis -
        the risk of running ACF/PACF or stationarity tests on the wrong column
        is too high.
      </div>
      <div class="ts-disabled-instructions">
        <div class="ts-inst-label">To enable, add the following to your config:</div>
        <pre class="ts-code-block">tasks:
  time_series:
    enabled: true
    datetime_index_column: your_date_column  # required</pre>
        <div class="ts-inst-label ts-inst-optional">
          Optional - leave blank to analyse all continuous columns:
        </div>
        <pre class="ts-code-block">    value_columns: []     # empty = all continuous columns
    frequency: null       # null = inferred from data</pre>
      </div>
      <div class="ts-disabled-note">
        You can also configure this via
        <strong>Settings → Time Series</strong> in the dashboard
        (coming soon), or by adding a
        <code>user_config.yaml</code> in your project root.
      </div>
    </div>

    <!-- ── No index column state ─────────────────────────────────────────── -->
    <div v-else-if="tsState === 'no_index'" class="card ts-disabled">
      <div class="ts-disabled-icon">⚠</div>
      <div class="ts-disabled-title">Time Series Index Column Not Set</div>
      <div class="ts-disabled-body">
        Time series is enabled but
        <code>time_series.datetime_index_column</code> is not configured
        or the column was not found in the dataset.
      </div>
      <div v-if="tsDisabledMessage" class="ts-error-detail">
        {{ tsDisabledMessage }}
      </div>
      <pre class="ts-code-block">tasks:
  time_series:
    enabled: true
    datetime_index_column: your_date_column</pre>
    </div>

    <!-- ── Ready ─────────────────────────────────────────────────────────── -->
    <template v-else>

      <!-- Header strip -->
      <div class="ts-header-strip card">
        <div class="ts-header-item">
          <span class="ts-header-label">Index Column</span>
          <span class="ts-header-value">{{ indexColumn }}</span>
        </div>
        <div class="ts-header-item" v-if="frequency">
          <span class="ts-header-label">Frequency</span>
          <span class="ts-header-value">{{ frequency }}</span>
        </div>
        <div class="ts-header-item">
          <span class="ts-header-label">Value Columns</span>
          <span class="ts-header-value">{{ valueColumnCount }}</span>
        </div>
      </div>

      <!-- ── 1. ACF / PACF ─────────────────────────────────────────────── -->
      <div class="card ts-section">
        <div class="ts-section-header" @click="toggleSection('acf')">
          <div class="ts-section-title">
            ACF / PACF
            <span class="ts-section-subtitle">Autocorrelation &amp; Partial Autocorrelation</span>
          </div>
          <button class="ts-collapse-btn">{{ openSections.has('acf') ? '▲' : '▼' }}</button>
        </div>

        <Transition name="ts-expand">
          <div v-if="openSections.has('acf')" class="ts-section-body">

            <div v-if="acfState === 'not_run'" class="es-not-run">
              ACF/PACF analysis did not run. Check that
              <code>time_series.enabled: true</code> and
              <code>datetime_index_column</code> is set in config.
            </div>
            <div v-else-if="acfState === 'error'" class="es-error">
              <span>⚠</span> {{ acfError }}
            </div>
            <div v-else-if="acfState === 'empty'" class="es-empty">
              ✓ No value columns produced ACF/PACF results.
            </div>
            <template v-else>
              <!-- Column selector -->
              <div class="ts-col-tabs">
                <button
                  v-for="name in Object.keys(acfData)"
                  :key="name"
                  class="ts-col-tab"
                  :class="{ 'ts-col-tab--active': acfSelectedCol === name }"
                  @click="acfSelectedCol = name"
                >{{ name }}</button>
              </div>

              <!-- ACF / PACF bar charts for selected column -->
              <div v-if="acfSelectedColData" class="ts-acf-charts">
                <div
                  v-for="kind in ['acf', 'pacf']"
                  :key="kind"
                  class="ts-acf-chart-wrap"
                >
                  <div class="ts-acf-chart-label">{{ kind.toUpperCase() }}</div>
                  <svg
                    class="ts-acf-svg"
                    :viewBox="`0 0 ${acfSvgW} ${acfSvgH}`"
                    preserveAspectRatio="none"
                  >
                    <!-- Zero baseline -->
                    <line
                      :x1="acfPadL" :y1="acfZeroY"
                      :x2="acfSvgW - acfPadR" :y2="acfZeroY"
                      class="ts-acf-baseline"
                    />
                    <!-- Confidence band (upper) -->
                    <line
                      :x1="acfPadL" :y1="acfConfY(kind, 'upper')"
                      :x2="acfSvgW - acfPadR" :y2="acfConfY(kind, 'upper')"
                      class="ts-acf-conf-line"
                    />
                    <!-- Confidence band (lower) -->
                    <line
                      :x1="acfPadL" :y1="acfConfY(kind, 'lower')"
                      :x2="acfSvgW - acfPadR" :y2="acfConfY(kind, 'lower')"
                      class="ts-acf-conf-line"
                    />
                    <!-- Confidence band fill -->
                    <rect
                      :x="acfPadL"
                      :y="acfConfY(kind, 'upper')"
                      :width="acfSvgW - acfPadL - acfPadR"
                      :height="Math.abs(acfConfY(kind, 'lower') - acfConfY(kind, 'upper'))"
                      class="ts-acf-conf-fill"
                    />
                    <!-- Bars -->
                    <rect
                      v-for="(lag, i) in acfSelectedColData.lags"
                      :key="i"
                      :x="acfBarX(i)"
                      :y="acfBarY(kind, i)"
                      :width="acfBarW"
                      :height="acfBarH(kind, i)"
                      :class="acfBarClass(kind, lag)"
                    />
                    <!-- Y axis labels -->
                    <text :x="acfPadL - 4" :y="acfZeroY + 4" class="ts-acf-axis-label ts-acf-axis-label--right">0</text>
                    <text :x="acfPadL - 4" :y="acfPadT + 4" class="ts-acf-axis-label ts-acf-axis-label--right">1</text>
                    <text :x="acfPadL - 4" :y="acfSvgH - acfPadB + 4" class="ts-acf-axis-label ts-acf-axis-label--right">−1</text>
                    <!-- X axis: first and last lag labels -->
                    <text
                      :x="acfBarX(0) + acfBarW / 2"
                      :y="acfSvgH - acfPadB + 12"
                      class="ts-acf-axis-label"
                    >1</text>
                    <text
                      :x="acfBarX(acfSelectedColData.lags.length - 1) + acfBarW / 2"
                      :y="acfSvgH - acfPadB + 12"
                      class="ts-acf-axis-label"
                    >{{ acfSelectedColData.lags[acfSelectedColData.lags.length - 1] }}</text>
                  </svg>
                  <div class="ts-acf-sig-note">
                    <span :class="acfSelectedColData[`${kind}_significant_lags`]?.length ? 'ts-flag' : 'ts-ok'">
                      {{ acfSelectedColData[`${kind}_significant_lags`]?.length ?? 0 }}
                    </span>
                    significant lag{{ (acfSelectedColData[`${kind}_significant_lags`]?.length ?? 0) !== 1 ? 's' : '' }}
                    <span v-if="acfSelectedColData[`${kind}_significant_lags`]?.length" class="ts-muted">
                      ({{ acfSelectedColData[`${kind}_significant_lags`].join(', ') }})
                    </span>
                  </div>
                </div>
              </div>

              <!-- Summary table -->
              <div class="ts-table">
                <div class="ts-table-row ts-table-header">
                  <span>Column</span>
                  <span>n</span>
                  <span>Sig. ACF lags</span>
                  <span>Sig. PACF lags</span>
                  <span>Pattern</span>
                </div>
                <div
                  v-for="(col, name) in acfData"
                  :key="name"
                  class="ts-table-row"
                  :class="{ 'ts-table-row--active': acfSelectedCol === name }"
                  @click="acfSelectedCol = name"
                  style="cursor:pointer"
                >
                  <span class="ts-col-name">{{ name }}</span>
                  <span class="ts-muted">{{ col.n?.toLocaleString() }}</span>
                  <span :class="col.acf_significant_lags?.length ? 'ts-flag' : 'ts-ok'">
                    {{ col.acf_significant_lags?.length ?? 0 }}
                  </span>
                  <span :class="col.pacf_significant_lags?.length ? 'ts-flag' : 'ts-ok'">
                    {{ col.pacf_significant_lags?.length ?? 0 }}
                  </span>
                  <span class="ts-muted ts-pattern">{{ acfPattern(col) }}</span>
                </div>
              </div>
            </template>
          </div>
        </Transition>
      </div>

      <!-- ── 2. Stationarity ────────────────────────────────────────────── -->
      <div class="card ts-section">
        <div class="ts-section-header" @click="toggleSection('stationarity')">
          <div class="ts-section-title">
            Stationarity Tests
            <span class="ts-section-subtitle">ADF (H₀: unit root) · KPSS (H₀: stationary)</span>
          </div>
          <button class="ts-collapse-btn">{{ openSections.has('stationarity') ? '▲' : '▼' }}</button>
        </div>

        <Transition name="ts-expand">
          <div v-if="openSections.has('stationarity')" class="ts-section-body">

            <div v-if="statState === 'not_run'" class="es-not-run">
              Stationarity tests did not run.
            </div>
            <div v-else-if="statState === 'error'" class="es-error">
              <span>⚠</span> {{ statError }}
            </div>
            <div v-else-if="statState === 'empty'" class="es-empty">
              ✓ No value columns tested (all may have been too short).
            </div>
            <template v-else>

              <!-- Epistemic note -->
              <div class="ts-epistemic-note">
                <span>ⓘ</span>
                {{ statEpistemicNote }}
              </div>

              <div class="ts-table">
                <div class="ts-table-row ts-table-header">
                  <span>Column</span>
                  <span>n</span>
                  <span>Assessment</span>
                  <span>Confidence</span>
                  <span>ADF p</span>
                  <span>KPSS p</span>
                </div>
                <div
                  v-for="(col, name) in statData"
                  :key="name"
                  class="ts-table-row"
                >
                  <span class="ts-col-name">{{ name }}</span>
                  <span class="ts-muted">{{ col.n?.toLocaleString() }}</span>
                  <span>
                    <span class="ts-assessment-badge" :class="assessmentClass(col.assessment)">
                      {{ assessmentLabel(col.assessment) }}
                    </span>
                  </span>
                  <span class="ts-muted">{{ col.confidence ?? '-' }}</span>
                  <span class="ts-muted">
                    {{ col.adf?.p_value != null ? col.adf.p_value.toFixed(4) : '-' }}
                  </span>
                  <span class="ts-muted">
                    {{ col.kpss?.p_value != null ? col.kpss.p_value.toFixed(4) : '-' }}
                  </span>
                </div>
              </div>

              <!-- Per-column caveats (collapsed by default) -->
              <div class="ts-caveats-toggle" @click="showCaveats = !showCaveats">
                {{ showCaveats ? '▲ Hide' : '▼ Show' }} test caveats
              </div>
              <div v-if="showCaveats" class="ts-caveats-block">
                <div v-for="(col, name) in statData" :key="name" class="ts-caveat-col">
                  <div class="ts-caveat-col-name">{{ name }}</div>
                  <ul class="ts-caveat-list">
                    <li v-for="(cav, i) in col.caveats ?? []" :key="i">{{ cav }}</li>
                  </ul>
                </div>
              </div>
            </template>

          </div>
        </Transition>
      </div>

      <!-- ── 3. STL Decomposition ───────────────────────────────────────── -->
      <div class="card ts-section">
        <div class="ts-section-header" @click="toggleSection('stl')">
          <div class="ts-section-title">
            STL Decomposition
            <span class="ts-section-subtitle">Trend · Seasonal · Residual</span>
          </div>
          <button class="ts-collapse-btn">{{ openSections.has('stl') ? '▲' : '▼' }}</button>
        </div>

        <Transition name="ts-expand">
          <div v-if="openSections.has('stl')" class="ts-section-body">

            <div v-if="stlState === 'not_run'" class="es-not-run">
              STL decomposition did not run.
            </div>
            <div v-else-if="stlState === 'error'" class="es-error">
              <span>⚠</span> {{ stlError }}
            </div>
            <div v-else-if="stlState === 'empty'" class="es-empty">
              <template v-if="stlEmptyMessage">
                <span class="ts-stl-empty-msg">{{ stlEmptyMessage }}</span>
              </template>
              <template v-else>
                ✓ No columns decomposed - series may be too short for the
                configured seasonal period, or no seasonal period could be inferred.
              </template>
              <span class="ts-empty-hint">
                Set <code>tasks.decompose_time_series.seasonal_period</code> in config
                to override auto-detection (e.g. <code>7</code> for daily/weekly,
                <code>12</code> for monthly/annual).
              </span>
            </div>
            <template v-else>
              <div class="ts-stl-meta">
                Seasonal period: <strong>{{ stlPeriod }}</strong> ·
                Robust: <strong>{{ stlRobust ? 'yes' : 'no' }}</strong>
              </div>

              <!-- STL 4-panel sparklines per column -->
              <div
                v-for="(col, name) in stlData"
                :key="name"
                class="ts-stl-column"
              >
                <div class="ts-stl-col-header">
                  <span class="ts-col-name">{{ name }}</span>
                  <span class="ts-muted ts-stl-n">n={{ col.n?.toLocaleString() }}</span>
                  <span class="ts-strength-pill ts-strength-pill--trend"
                    :class="strengthClass(col.trend_strength)">
                    Trend {{ strengthLabel(col.trend_strength) }}
                  </span>
                  <span class="ts-strength-pill ts-strength-pill--seasonal"
                    :class="strengthClass(col.seasonal_strength)">
                    Seasonal {{ strengthLabel(col.seasonal_strength) }}
                  </span>
                </div>
                <div class="ts-stl-panels">
                  <div
                    v-for="panel in ['observed', 'trend', 'seasonal', 'residual']"
                    :key="panel"
                    class="ts-stl-panel"
                  >
                    <div class="ts-stl-panel-label">{{ panel }}</div>
                    <svg
                      class="ts-stl-svg"
                      :viewBox="`0 0 ${stlSvgW} ${stlSvgH}`"
                      preserveAspectRatio="none"
                    >
                      <!-- Zero line for seasonal/residual -->
                      <line
                        v-if="panel === 'seasonal' || panel === 'residual'"
                        :x1="0" :y1="stlZeroY(col, panel)"
                        :x2="stlSvgW" :y2="stlZeroY(col, panel)"
                        class="ts-stl-zeroline"
                      />
                      <!-- Sparkline -->
                      <polyline
                        :points="stlPoints(col, panel)"
                        :class="`ts-stl-line ts-stl-line--${panel}`"
                        fill="none"
                      />
                    </svg>
                    <div class="ts-stl-panel-range">
                      <span>{{ stlPanelMin(col, panel) }}</span>
                      <span>{{ stlPanelMax(col, panel) }}</span>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Strength summary table -->
              <div class="ts-table">
                <div class="ts-table-row ts-table-header">
                  <span>Column</span>
                  <span>n</span>
                  <span>Trend strength</span>
                  <span>Seasonal strength</span>
                </div>
                <div
                  v-for="(col, name) in stlData"
                  :key="name"
                  class="ts-table-row"
                >
                  <span class="ts-col-name">{{ name }}</span>
                  <span class="ts-muted">{{ col.n?.toLocaleString() }}</span>
                  <span>
                    <span class="ts-strength-bar-wrap">
                      <span
                        class="ts-strength-bar ts-strength-bar--trend"
                        :style="{ width: `${(col.trend_strength ?? 0) * 100}%` }"
                      />
                    </span>
                    <span
                      class="ts-strength-val"
                      :class="strengthClass(col.trend_strength)"
                    >{{ strengthLabel(col.trend_strength) }}</span>
                  </span>
                  <span>
                    <span class="ts-strength-bar-wrap">
                      <span
                        class="ts-strength-bar ts-strength-bar--seasonal"
                        :style="{ width: `${(col.seasonal_strength ?? 0) * 100}%` }"
                      />
                    </span>
                    <span
                      class="ts-strength-val"
                      :class="strengthClass(col.seasonal_strength)"
                    >{{ strengthLabel(col.seasonal_strength) }}</span>
                  </span>
                </div>
              </div>
            </template>

          </div>
        </Transition>
      </div>

    </template>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  tasks: { type: Object, default: () => ({}) },
  theme: { type: String, default: 'dark' },
})

// ── Section collapse ──────────────────────────────────────────────────────────
const openSections = ref(new Set(['acf', 'stationarity', 'stl']))
const showCaveats  = ref(false)

function toggleSection(key) {
  const s = new Set(openSections.value)
  s.has(key) ? s.delete(key) : s.add(key)
  openSections.value = s
}

// ── Top-level TS state ────────────────────────────────────────────────────────
// Determine from the summary message of any TS task whether TS is disabled
// or simply hasn't produced results.

function tsTaskSummary(name) {
  return props.tasks?.[name]?.summary ?? null
}

const tsState = computed(() => {
  const acfSummary  = tsTaskSummary('compute_acf_pacf')
  const statSummary = tsTaskSummary('detect_stationarity')
  const stlSummary  = tsTaskSummary('decompose_time_series')

  // No TS tasks in the payload at all — tasks didn't run or weren't persisted
  if (!acfSummary && !statSummary && !stlSummary) return 'disabled'

  const summaries = [acfSummary, statSummary, stlSummary].filter(Boolean)

  // Check if ALL tasks reported disabled (TS globally off or no index column)
  const allDisabled = summaries.every(s => s.time_series_enabled === false)
  if (allDisabled) {
    const msg = summaries[0]?.message ?? ''
    const msgLower = msg.toLowerCase()
    if (
      msgLower.includes('index_column') ||
      msgLower.includes('datetime_index') ||
      msgLower.includes('not present') ||
      msgLower.includes('not found') ||
      msgLower.includes('not set')
    ) {
      return 'no_index'
    }
    return 'disabled'
  }

  // At least one task produced real results → show the ready state.
  // Individual tasks that failed (e.g. decompose with no seasonal period)
  // are handled per-section with their own error/empty states.
  return 'ready'
})

// The message from a disabled TS task — shown in the no_index state for context.
const tsDisabledMessage = computed(() => {
  const summaries = [
    tsTaskSummary('compute_acf_pacf'),
    tsTaskSummary('detect_stationarity'),
    tsTaskSummary('decompose_time_series'),
  ].filter(Boolean)
  return summaries.find(s => s.time_series_enabled === false)?.message ?? null
})

// ── Shared metadata ───────────────────────────────────────────────────────────
const indexColumn    = computed(() =>
  props.tasks?.compute_acf_pacf?.metadata?.index_column ??
  props.tasks?.detect_stationarity?.metadata?.index_column ??
  props.tasks?.decompose_time_series?.metadata?.index_column ?? '-'
)
const frequency      = computed(() =>
  props.tasks?.compute_acf_pacf?.metadata?.frequency ??
  props.tasks?.detect_stationarity?.metadata?.frequency ?? null
)
const valueColumnCount = computed(() => {
  const cols = Object.keys(acfData.value ?? {}).length ||
               Object.keys(statData.value ?? {}).length ||
               Object.keys(stlData.value ?? {}).length
  return cols || '-'
})

// ── ACF / PACF ────────────────────────────────────────────────────────────────
const acfResult = computed(() => props.tasks?.compute_acf_pacf ?? null)
const acfState  = computed(() => {
  if (!acfResult.value)                          return 'not_run'
  if (acfResult.value.status === 'error')        return 'error'
  if (acfResult.value.summary?.time_series_enabled === false) return 'not_run'
  if (!Object.keys(acfResult.value.data ?? {}).length) return 'empty'
  return 'ready'
})
const acfError  = computed(() =>
  acfResult.value?.error_metadata?.message ?? 'ACF/PACF task encountered an error.'
)
const acfData   = computed(() => acfResult.value?.data ?? {})

function acfPattern(col) {
  const acfSig  = col.acf_significant_lags?.length ?? 0
  const pacfSig = col.pacf_significant_lags?.length ?? 0
  if (!acfSig && !pacfSig) return 'No autocorrelation'
  if (pacfSig && !acfSig)  return `AR(${Math.max(...col.pacf_significant_lags)})`
  if (acfSig && !pacfSig)  return `MA(${Math.max(...col.acf_significant_lags)})`
  return 'ARMA'
}

// ── Stationarity ──────────────────────────────────────────────────────────────
const statResult = computed(() => props.tasks?.detect_stationarity ?? null)
const statState  = computed(() => {
  if (!statResult.value)                          return 'not_run'
  if (statResult.value.status === 'error')        return 'error'
  if (statResult.value.summary?.time_series_enabled === false) return 'not_run'
  if (!Object.keys(statResult.value.data ?? {}).length) return 'empty'
  return 'ready'
})
const statError  = computed(() =>
  statResult.value?.error_metadata?.message ?? 'Stationarity task encountered an error.'
)
const statData   = computed(() => statResult.value?.data ?? {})
const statEpistemicNote = computed(() =>
  statResult.value?.summary?.epistemic_note ??
  'ADF and KPSS have opposite null hypotheses. Agreement between them ' +
  'strengthens the conclusion; disagreement signals ambiguity.'
)

function assessmentLabel(assessment) {
  const labels = {
    consistent_with_stationary:           'Consistent with stationary',
    consistent_with_non_stationary:       'Consistent with non-stationary',
    ambiguous_trend_stationary_or_structural_break: 'Ambiguous',
    inconclusive:                          'Inconclusive',
    indeterminate:                         'Indeterminate',
  }
  return labels[assessment] ?? (assessment?.replace(/_/g, ' ') ?? '-')
}

function assessmentClass(assessment) {
  if (!assessment) return 'ts-badge--gray'
  if (assessment === 'consistent_with_stationary')     return 'ts-badge--green'
  if (assessment === 'consistent_with_non_stationary') return 'ts-badge--red'
  return 'ts-badge--amber'
}

// ── STL Decomposition ─────────────────────────────────────────────────────────
const stlResult = computed(() => props.tasks?.decompose_time_series ?? null)
const stlState  = computed(() => {
  if (!stlResult.value)                          return 'not_run'
  if (stlResult.value.status === 'error')        return 'error'
  // time_series_enabled: false on STL specifically means it couldn't determine
  // the seasonal period — show as empty with a config hint, not 'not_run'
  if (stlResult.value.summary?.time_series_enabled === false) return 'empty'
  if (!Object.keys(stlResult.value.data ?? {}).length) return 'empty'
  return 'ready'
})
const stlError  = computed(() =>
  stlResult.value?.error_metadata?.message ?? 'STL decomposition task encountered an error.'
)
// The message explaining why STL is empty (e.g. no seasonal period detected)
const stlEmptyMessage = computed(() =>
  stlResult.value?.summary?.time_series_enabled === false
    ? stlResult.value?.summary?.message ?? null
    : null
)
const stlData   = computed(() => stlResult.value?.data ?? {})
const stlPeriod = computed(() => stlResult.value?.summary?.seasonal_period ?? '-')
const stlRobust = computed(() => stlResult.value?.summary?.robust ?? true)

// ── ACF / PACF chart ──────────────────────────────────────────────────────────
// Column selector state — auto-select first column when data arrives
const acfSelectedCol = ref(null)
const acfSelectedColData = computed(() => {
  const cols = Object.keys(acfData.value)
  if (!acfSelectedCol.value && cols.length) acfSelectedCol.value = cols[0]
  return acfData.value[acfSelectedCol.value] ?? null
})

// SVG dimensions
const acfSvgW = 340
const acfSvgH = 120
const acfPadL = 28
const acfPadR = 8
const acfPadT = 8
const acfPadB = 16

// Y scale: maps [-1, 1] → [padT, svgH - padB]
function acfYScale(v) {
  const h = acfSvgH - acfPadT - acfPadB
  return acfPadT + h * (1 - (v + 1) / 2)
}
const acfZeroY = computed(() => acfYScale(0))

function acfConfY(kind, bound) {
  const d = acfSelectedColData.value
  if (!d) return acfZeroY.value
  const arr = d[`${kind}_confidence_${bound}`]
  if (!arr?.length) return acfZeroY.value
  // Use the mean of the confidence band (it's approximately constant)
  const mean = arr.reduce((s, v) => s + Math.abs(v), 0) / arr.length
  return bound === 'upper' ? acfYScale(mean) : acfYScale(-mean)
}

const acfBarW = computed(() => {
  const d = acfSelectedColData.value
  if (!d?.lags?.length) return 4
  const totalW = acfSvgW - acfPadL - acfPadR
  return Math.max(1, Math.min(8, totalW / d.lags.length - 1))
})

function acfBarX(i) {
  const d = acfSelectedColData.value
  if (!d?.lags?.length) return acfPadL
  const totalW = acfSvgW - acfPadL - acfPadR
  const step = totalW / d.lags.length
  return acfPadL + i * step + (step - acfBarW.value) / 2
}

function acfBarY(kind, i) {
  const d = acfSelectedColData.value
  const v = d?.[`${kind}_values`]?.[i] ?? 0
  return v >= 0 ? acfYScale(v) : acfZeroY.value
}

function acfBarH(kind, i) {
  const d = acfSelectedColData.value
  const v = d?.[`${kind}_values`]?.[i] ?? 0
  return Math.abs(acfYScale(v) - acfZeroY.value)
}

function acfBarClass(kind, lag) {
  const d = acfSelectedColData.value
  const sigLags = d?.[`${kind}_significant_lags`] ?? []
  return sigLags.includes(lag) ? 'ts-acf-bar ts-acf-bar--sig' : 'ts-acf-bar'
}

// ── STL sparkline chart ───────────────────────────────────────────────────────
const stlSvgW = 200
const stlSvgH = 48
const stlPad  = 4

function stlValues(col, panel) {
  return col[panel] ?? []
}

function stlMinMax(col, panel) {
  const vals = stlValues(col, panel)
  if (!vals.length) return [0, 1]
  const mn = Math.min(...vals)
  const mx = Math.max(...vals)
  return mn === mx ? [mn - 1, mn + 1] : [mn, mx]
}

function stlPanelMin(col, panel) {
  return stlMinMax(col, panel)[0].toFixed(3)
}
function stlPanelMax(col, panel) {
  return stlMinMax(col, panel)[1].toFixed(3)
}

function stlZeroY(col, panel) {
  const [mn, mx] = stlMinMax(col, panel)
  const h = stlSvgH - stlPad * 2
  return stlPad + h * (1 - (0 - mn) / (mx - mn))
}

function stlPoints(col, panel) {
  const vals = stlValues(col, panel)
  if (!vals.length) return ''
  const [mn, mx] = stlMinMax(col, panel)
  const w = stlSvgW - stlPad * 2
  const h = stlSvgH - stlPad * 2
  return vals.map((v, i) => {
    const x = stlPad + (i / (vals.length - 1)) * w
    const y = stlPad + h * (1 - (v - mn) / (mx - mn))
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function strengthLabel(v) {
  if (v == null) return '-'
  if (v >= 0.7)  return `Strong (${v.toFixed(2)})`
  if (v >= 0.4)  return `Moderate (${v.toFixed(2)})`
  if (v >= 0.1)  return `Weak (${v.toFixed(2)})`
  return `Negligible (${v.toFixed(2)})`
}

function strengthClass(v) {
  if (v == null || v < 0.1) return 'ts-strength--low'
  if (v >= 0.7)              return 'ts-strength--high'
  if (v >= 0.4)              return 'ts-strength--mid'
  return 'ts-strength--low'
}
</script>

<style scoped>
.ts-tab { display: flex; flex-direction: column; gap: 16px; }

/* ── Disabled / empty state card ─────────────────────────────────────────── */
.ts-disabled {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 48px 40px;
  text-align: center;
  max-width: 680px;
  margin: 0 auto;
  width: 100%;
}

.ts-disabled-icon  { font-size: 40px; opacity: 0.5; }
.ts-disabled-title { font-size: 18px; font-weight: 600; color: #e2e8f0; }
.ts-disabled-body  { font-size: 13px; color: #64748b; line-height: 1.6; max-width: 520px; }

.ts-disabled-instructions {
  width: 100%;
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.ts-inst-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #64748b;
}
.ts-inst-optional { margin-top: 8px; }

.ts-code-block {
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 6px;
  padding: 12px 16px;
  font-size: 12px;
  font-family: ui-monospace, 'Cascadia Code', monospace;
  color: #93c5fd;
  line-height: 1.6;
  margin: 0;
  text-align: left;
  white-space: pre;
  overflow-x: auto;
}

.ts-disabled-note {
  font-size: 12px;
  color: #64748b;
  line-height: 1.5;
}
.ts-disabled-note strong { color: #94a3b8; }
.ts-disabled-note code   { color: #93c5fd; font-size: 11px; }

.ts-error-detail {
  width: 100%;
  text-align: left;
  background: #1a0f0f;
  border: 1px solid #7f1d1d;
  border-left: 3px solid #f87171;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 12px;
  color: #fca5a5;
  line-height: 1.5;
  font-family: ui-monospace, 'Cascadia Code', monospace;
}

/* ── Header strip ────────────────────────────────────────────────────────── */
.ts-header-strip {
  display: flex;
  gap: 0;
  padding: 14px 24px;
}

.ts-header-item {
  display: flex;
  flex-direction: column;
  gap: 3px;
  flex: 1;
  padding-right: 24px;
  border-right: 1px solid #334155;
  margin-right: 24px;
}
.ts-header-item:last-child { border-right: none; margin-right: 0; padding-right: 0; }

.ts-header-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #64748b;
}
.ts-header-value {
  font-size: 15px;
  font-weight: 600;
  color: #f1f5f9;
  font-family: ui-monospace, 'Cascadia Code', monospace;
}

/* ── Section cards ───────────────────────────────────────────────────────── */
.ts-section { padding: 0; overflow: visible; }

.ts-section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  cursor: pointer;
  user-select: none;
  transition: background 0.12s;
}
.ts-section-header:hover { background: rgba(255,255,255,0.03); }

.ts-section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  font-weight: 600;
  color: #e2e8f0;
}
.ts-section-subtitle {
  font-size: 11px;
  font-weight: 400;
  color: #64748b;
}

.ts-collapse-btn {
  padding: 3px 8px;
  font-size: 11px;
  background: none;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #64748b;
  cursor: pointer;
  transition: all 0.12s;
}
.ts-collapse-btn:hover { border-color: #60a5fa; color: #93c5fd; }

/* ── Section body ────────────────────────────────────────────────────────── */
.ts-section-body {
  border-top: 1px solid #1e293b;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

/* ── Empty / error states ────────────────────────────────────────────────── */
.es-not-run { color: #64748b; font-size: 13px; padding: 16px 0; text-align: center; }
.es-empty   { color: #4ade80; font-size: 13px; padding: 16px 0; text-align: center; display: flex; flex-direction: column; gap: 6px; align-items: center; }
.ts-empty-hint { font-size: 11px; color: #64748b; }
.ts-empty-hint code { color: #93c5fd; font-size: 11px; }
.es-error   { color: #f87171; font-size: 13px; padding: 12px; background: #3d0f0f; border-radius: 6px; border-left: 3px solid #f87171; display: flex; gap: 8px; }

/* ── ACF / PACF chart ────────────────────────────────────────────────────── */
.ts-col-tabs {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 4px;
}
.ts-col-tab {
  padding: 3px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-family: ui-monospace, 'Cascadia Code', monospace;
  background: #1e293b;
  border: 1px solid #334155;
  color: #64748b;
  cursor: pointer;
  transition: all 0.12s;
}
.ts-col-tab:hover { border-color: #60a5fa; color: #93c5fd; }
.ts-col-tab--active { background: #0f2035; border-color: #60a5fa; color: #93c5fd; }

.ts-acf-charts {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.ts-acf-chart-wrap {
  flex: 1;
  min-width: 200px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.ts-acf-chart-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: #64748b;
  text-transform: uppercase;
}
.ts-acf-svg {
  width: 100%;
  height: 120px;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 4px;
  overflow: visible;
}
.ts-acf-baseline { stroke: #334155; stroke-width: 1; }
.ts-acf-conf-line { stroke: #1e3a5f; stroke-width: 1; stroke-dasharray: 3 2; }
.ts-acf-conf-fill { fill: #1e3a5f; opacity: 0.25; }
.ts-acf-bar { fill: #3b5998; }
.ts-acf-bar--sig { fill: #60a5fa; }
.ts-acf-axis-label {
  font-size: 8px;
  fill: #475569;
  text-anchor: middle;
  font-family: ui-monospace, monospace;
}
.ts-acf-axis-label--right { text-anchor: end; }
.ts-acf-sig-note { font-size: 11px; color: #64748b; }

.ts-table-row--active { background: rgba(96, 165, 250, 0.05); }

/* ── STL sparklines ──────────────────────────────────────────────────────── */
.ts-stl-column {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px 0;
  border-bottom: 1px solid #1e293b;
}
.ts-stl-column:last-of-type { border-bottom: none; }

.ts-stl-col-header {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.ts-stl-n { font-size: 11px; }

.ts-strength-pill {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 9px;
  font-size: 10px;
  font-weight: 600;
  border: 1px solid;
}
.ts-strength-pill--trend   { border-color: #3b82f6; color: #60a5fa; background: #0f2035; }
.ts-strength-pill--seasonal { border-color: #8b5cf6; color: #a78bfa; background: #1a1035; }

.ts-stl-panels {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}
.ts-stl-panel {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.ts-stl-panel-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #64748b;
}
.ts-stl-svg {
  width: 100%;
  height: 48px;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 3px;
  overflow: hidden;
}
.ts-stl-zeroline { stroke: #334155; stroke-width: 1; stroke-dasharray: 2 2; }
.ts-stl-line { stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.ts-stl-line--observed { stroke: #e2e8f0; }
.ts-stl-line--trend    { stroke: #60a5fa; }
.ts-stl-line--seasonal { stroke: #a78bfa; }
.ts-stl-line--residual { stroke: #94a3b8; }
.ts-stl-panel-range {
  display: flex;
  justify-content: space-between;
  font-size: 9px;
  color: #475569;
  font-family: ui-monospace, monospace;
}

/* ── Data tables ─────────────────────────────────────────────────────────── */
.ts-table { width: 100%; }

.ts-table-row {
  display: grid;
  grid-template-columns: 160px 80px 1fr 1fr 80px 80px;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid #0f172a;
  font-size: 13px;
}
.ts-table-row:last-child { border-bottom: none; }
.ts-table-row:hover:not(.ts-table-header) { background: rgba(255,255,255,0.02); }

.ts-table-header {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
  border-bottom: 1px solid #334155;
  padding-bottom: 8px;
}

/* STL table has different column layout */
.ts-section:last-child .ts-table-row {
  grid-template-columns: 160px 80px 1fr 1fr;
}

.ts-col-name {
  font-family: ui-monospace, 'Cascadia Code', monospace;
  font-size: 12px;
  color: #93c5fd;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ts-muted   { color: #64748b; font-size: 12px; }
.ts-flag    { color: #fbbf24; font-weight: 600; }
.ts-ok      { color: #4ade80; }
.ts-pattern { font-size: 11px; }

/* ── Assessment badges ───────────────────────────────────────────────────── */
.ts-assessment-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 9px;
  font-size: 11px;
  font-weight: 600;
  border: 1px solid;
  white-space: nowrap;
}
.ts-badge--green { background: #0f2718; color: #4ade80; border-color: #4ade80; }
.ts-badge--red   { background: #3d0f0f; color: #f87171; border-color: #f87171; }
.ts-badge--amber { background: #3d2a00; color: #fbbf24; border-color: #fbbf24; }
.ts-badge--gray  { background: #1e293b; color: #94a3b8; border-color: #64748b; }

/* ── Epistemic note ──────────────────────────────────────────────────────── */
.ts-epistemic-note {
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

/* ── Caveats ─────────────────────────────────────────────────────────────── */
.ts-caveats-toggle {
  font-size: 11px;
  color: #64748b;
  cursor: pointer;
  user-select: none;
  padding: 4px 0;
}
.ts-caveats-toggle:hover { color: #94a3b8; }

.ts-caveats-block {
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: #0f172a;
  border-radius: 6px;
  padding: 12px 16px;
}
.ts-caveat-col-name {
  font-family: ui-monospace, 'Cascadia Code', monospace;
  font-size: 12px;
  color: #93c5fd;
  margin-bottom: 4px;
}
.ts-caveat-list {
  margin: 0;
  padding-left: 16px;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.ts-caveat-list li { font-size: 11px; color: #64748b; line-height: 1.4; }

/* ── STL strength bars ───────────────────────────────────────────────────── */
.ts-stl-meta { font-size: 12px; color: #64748b; }
.ts-stl-meta strong { color: #e2e8f0; }

.ts-strength-bar-wrap {
  display: inline-block;
  width: 60px;
  height: 6px;
  background: #1e293b;
  border-radius: 3px;
  overflow: hidden;
  vertical-align: middle;
  margin-right: 8px;
}
.ts-strength-bar {
  display: block;
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s;
}
.ts-strength-bar--trend   { background: #60a5fa; }
.ts-strength-bar--seasonal { background: #a78bfa; }

.ts-strength-val { font-size: 12px; }
.ts-strength--high { color: #f87171; font-weight: 600; }
.ts-strength--mid  { color: #fbbf24; }
.ts-strength--low  { color: #4ade80; }

/* ── Expand/collapse transition ──────────────────────────────────────────── */
.ts-expand-enter-active,
.ts-expand-leave-active {
  transition: opacity 0.18s, max-height 0.22s ease;
  max-height: 2000px;
  overflow: hidden;
}
.ts-expand-enter-from,
.ts-expand-leave-to { opacity: 0; max-height: 0; }

/* ── Light theme ─────────────────────────────────────────────────────────── */
:global(.theme-light) .ts-code-block          { background: #f8fafc; border-color: #e2e8f0; color: #2563eb; }
:global(.theme-light) .ts-section-header:hover { background: rgba(0,0,0,0.02); }
:global(.theme-light) .ts-section-body        { border-top-color: #e2e8f0; }
:global(.theme-light) .ts-table-row           { border-bottom-color: #f1f5f9; }
:global(.theme-light) .ts-table-header        { border-bottom-color: #e2e8f0; color: #94a3b8; }
:global(.theme-light) .ts-col-name            { color: #2563eb; }
:global(.theme-light) .ts-epistemic-note      { background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8; }
:global(.theme-light) .ts-caveats-block       { background: #f8fafc; }
:global(.theme-light) .ts-caveat-list li      { color: #94a3b8; }
:global(.theme-light) .ts-strength-bar-wrap   { background: #e2e8f0; }
:global(.theme-light) .ts-disabled-note       { color: #94a3b8; }
:global(.theme-light) .ts-header-item         { border-right-color: #e2e8f0; }
:global(.theme-light) .ts-header-value        { color: #1e293b; }
:global(.theme-light) .es-not-run             { color: #94a3b8; }
:global(.theme-light) .ts-col-tab             { background: #f1f5f9; border-color: #e2e8f0; color: #94a3b8; }
:global(.theme-light) .ts-col-tab--active     { background: #eff6ff; border-color: #3b82f6; color: #1d4ed8; }
:global(.theme-light) .ts-acf-svg            { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .ts-acf-baseline       { stroke: #e2e8f0; }
:global(.theme-light) .ts-acf-conf-line      { stroke: #bfdbfe; }
:global(.theme-light) .ts-acf-conf-fill      { fill: #bfdbfe; }
:global(.theme-light) .ts-acf-bar            { fill: #93c5fd; }
:global(.theme-light) .ts-acf-bar--sig       { fill: #2563eb; }
:global(.theme-light) .ts-acf-axis-label     { fill: #94a3b8; }
:global(.theme-light) .ts-stl-svg            { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .ts-stl-zeroline       { stroke: #e2e8f0; }
:global(.theme-light) .ts-stl-line--observed { stroke: #1e293b; }
:global(.theme-light) .ts-stl-panel-range    { color: #94a3b8; }
:global(.theme-light) .ts-stl-col-header .ts-col-name { color: #2563eb; }
:global(.theme-light) .ts-stl-column        { border-bottom-color: #f1f5f9; }
</style>
