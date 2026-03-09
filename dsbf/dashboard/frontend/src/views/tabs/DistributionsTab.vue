<template>
  <div class="distributions-tab">
    <!-- Left: column browser -->
    <div class="browser-pane card">
      <div class="card-title">Columns</div>
      <ColumnBrowser
        :tasks="tasks"
        :selected="selectedColumn"
        @select="selectedColumn = $event"
      />
    </div>

    <!-- Right: detail panel -->
    <div v-if="selectedColumn" class="detail-pane">
      <h2 class="col-heading">
        <span class="col-heading-name">{{ selectedColumn }}</span>
        <span class="col-heading-badge" :class="`intent-${columnIntent}`">
          {{ columnIntent }}
        </span>
        <span class="col-heading-dtype muted">{{ columnDtype }}</span>
      </h2>

      <!-- Section 1: Stats strip -->
      <ColumnStatsStrip :column="selectedColumn" :tasks="tasks" />

      <!-- Section 2: Plot + Guidance -->
      <div class="plot-guidance-row">
        <div class="plot-pane">
          <PlotCard
            v-if="columnFigure"
            :title="plotTitle"
            :figure="columnFigure"
            :theme="theme"
          >
            <template v-if="columnIntent === 'continuous'" #controls>
              <div class="plot-toggle">
                <button
                  v-for="opt in plotOptions"
                  :key="opt.key"
                  class="toggle-btn"
                  :class="{ active: activePlotType === opt.key }"
                  @click="activePlotType = opt.key"
                >{{ opt.label }}</button>
              </div>
            </template>
          </PlotCard>
          <div v-else class="card no-figure-card">
            <div class="card-title">{{ plotTitle }}</div>
            <div class="no-figure">No figure available for this column.</div>
          </div>
        </div>
        <div class="guidance-pane">
          <ColumnGuidanceCard :column="selectedColumn" :tasks="tasks" />
        </div>
      </div>

      <!-- Section 3: Context table -->
      <PercentilesTable
        v-if="columnIntent === 'continuous'"
        :column="selectedColumn"
        :tasks="tasks"
      />
      <ValueCountsTable
        v-else-if="showValueCounts"
        :column="selectedColumn"
        :tasks="tasks"
      />
      <TextLengthCard
        v-if="columnIntent === 'text' || columnIntent === 'categorical'"
        :column="selectedColumn"
        :tasks="tasks"
      />

    </div>

    <!-- Empty state -->
    <div v-else class="empty-state card">
      <span class="empty-icon">←</span>
      <p>Select a column from the browser to explore its distribution.</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import ColumnBrowser      from '../../components/distributions/ColumnBrowser.vue'
import ColumnStatsStrip   from '../../components/distributions/ColumnStatsStrip.vue'
import ColumnGuidanceCard from '../../components/distributions/ColumnGuidanceCard.vue'
import PercentilesTable        from '../../components/distributions/PercentilesTable.vue'
import ValueCountsTable        from '../../components/distributions/ValueCountsTable.vue'
import TextLengthCard          from '../../components/distributions/TextLengthCard.vue'
import PlotCard           from '../../components/overview/PlotCard.vue'
import { figureForColumn } from '../../utils.js'

const props = defineProps({
  run:     { type: Object, default: null },
  tasks:   { type: Object, default: () => ({}) },
  figures: { type: Array,  default: () => [] },
  theme:   { type: String, default: 'dark' },
})

const selectedColumn = ref(null)
const activePlotType = ref('histogram')

const plotOptions = [
  { key: 'histogram', label: 'Histogram' },
  { key: 'boxplot',   label: 'Box Plot' },
]

// Reset to histogram when switching columns
watch(selectedColumn, () => { activePlotType.value = 'histogram' })

const columnIntent = computed(() =>
  props.tasks.infer_types?.data?.[selectedColumn.value]?.analysis_intent_dtype ?? 'unknown'
)
const columnDtype = computed(() =>
  props.tasks.infer_types?.data?.[selectedColumn.value]?.inferred_dtype ?? ''
)

// Pick the best figure for the selected column:
// - numeric: prefer interactive histogram; fall back to static composite
// - categorical/boolean: prefer interactive bar; fall back to static bar
const columnFigure = computed(() => {
  const col    = selectedColumn.value
  const intent = columnIntent.value
  if (!col) return null

  if (intent === 'continuous') {
    const type = activePlotType.value  // 'histogram' or 'boxplot'
    return (
      figureForColumn(props.figures, col, type, 'interactive', props.theme) ??
      figureForColumn(props.figures, col, type, 'static',      props.theme) ??
      // fallback to composite static if specific type not found
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

const plotTitle = computed(() => {
  const intent = columnIntent.value
  if (intent === 'continuous') {
    return activePlotType.value === 'boxplot' ? 'Box Plot' : 'Distribution'
  }
  return 'Value Counts'
})

const showValueCounts = computed(() =>
  ['categorical', 'boolean', 'text', 'datetime', 'unknown'].includes(columnIntent.value)
)
</script>

<style scoped>
.distributions-tab {
  display: flex;
  gap: 16px;
  align-items: flex-start;
  min-height: 600px;
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
  gap: 16px;
}

.col-heading {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  color: #f1f5f9;
  flex-wrap: wrap;
}

.col-heading-name {
  font-family: monospace;
  font-size: 18px;
}

.col-heading-badge {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 3px 9px;
  border-radius: 9px;
  border: 1px solid;
}
.intent-continuous  { background: #1e3a5f; color: #60a5fa; border-color: #60a5fa; }
.intent-categorical { background: #14291f; color: #4ade80; border-color: #4ade80; }
.intent-boolean     { background: #2d1b4e; color: #c084fc; border-color: #c084fc; }
.intent-datetime    { background: #3d2510; color: #fb923c; border-color: #fb923c; }
.intent-text        { background: #3d0f29; color: #f472b6; border-color: #f472b6; }
.intent-unknown     { background: #1e293b; color: #94a3b8; border-color: #475569; }

.col-heading-dtype {
  font-size: 13px;
  font-family: monospace;
}
.muted { color: #475569; }

/* ── Section 2: plot + guidance ──────────────────────────────────────────── */
.plot-guidance-row {
  display: flex;
  gap: 16px;
  align-items: stretch;
  height: 480px;
}
.plot-pane {
  flex: 3 1 0;
  min-width: 0;
  height: 100%;
  overflow: hidden;
}
.guidance-pane {
  flex: 1 1 0;
  min-width: 220px;
  height: 100%;
  overflow: hidden;
}

/* Make PlotCard and GuidanceCard fill the fixed row height */
.plot-pane :deep(.plot-card),
.plot-pane :deep(.no-figure-card),
.guidance-pane :deep(.guidance-card) {
  height: 100%;
  overflow-y: auto;
}

/* Plotly container should fill available space inside the fixed card */
.plot-pane :deep(.plot-container) {
  min-height: 0;
  height: calc(100% - 32px); /* subtract card-title height */
}

.no-figure {
  color: #475569;
  font-size: 13px;
  padding: 40px 0;
  text-align: center;
}

/* ── Plot type toggle (rendered inside PlotCard controls slot) ───────────── */
.plot-pane :deep(.plot-toggle) {
  display: flex;
  gap: 4px;
}

.plot-pane :deep(.toggle-btn) {
  padding: 3px 12px;
  border-radius: 6px;
  border: 1px solid #334155;
  background: #0f172a;
  color: #64748b;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}
.plot-pane :deep(.toggle-btn:hover)  { background: #1e293b; color: #94a3b8; }
.plot-pane :deep(.toggle-btn.active) { background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }

/* ── Empty state ─────────────────────────────────────────────────────────── */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: #475569;
  min-height: 300px;
}
.empty-icon { font-size: 32px; }
.empty-state p { font-size: 14px; text-align: center; margin: 0; }
</style>
