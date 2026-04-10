<template>
  <div class="relationships-tab">

    <!-- ── Dataset-level overview (always visible) ────────────────────────── -->
    <RelationshipsSummaryCard
      :tasks="tasks"
      :active-filter="activeWarningFilter"
      @filter-warnings="activeWarningFilter = activeWarningFilter === $event ? null : $event"
    />

    <CorrelationHeatmapCard :tasks="tasks" />

    <!-- ── Column-pair exploration ────────────────────────────────────────── -->
    <div class="tab-main">

      <!-- Left: column browser -->
      <div class="browser-pane card">
        <div class="card-title">Columns</div>
        <ColumnBrowser
          :tasks="tasks"
          :selected="primaryColumn"
          :showNullBar="false"
          :warningSet="relationshipWarnings"
          alertLabel="Collinearity / leakage"
          @select="selectPrimary"
        />
      </div>

      <!-- Right: detail panel -->
      <div v-if="primaryColumn" class="detail-pane">

        <!-- Primary column heading -->
        <div class="col-heading">
          <span class="col-heading-name">{{ primaryColumn }}</span>
          <span class="col-heading-badge" :class="`intent-${primaryIntent}`">
            {{ primaryIntent }}
          </span>
        </div>

        <!-- Associations panel -->
        <div class="card assoc-panel">
          <div class="card-title-row">
            <span class="card-title">
              Associations
              <TooltipIcon
                text="Pairwise association between this column and all others. The metric used depends on the type combination: Pearson r (continuous × continuous), eta squared (continuous × categorical), Cramér's V (categorical × categorical), point-biserial r (continuous × boolean)."
                direction="down"
                align="left"
              />
            </span>
            <span class="pair-count" v-if="!assocLoading && !assocUnavailable">
              {{ assocRows.length }} columns
            </span>
          </div>

          <div v-if="assocLoading"     class="panel-state">Loading associations…</div>
          <div v-else-if="assocError"  class="panel-state error-text">{{ assocError }}</div>
          <div v-else-if="assocUnavailable" class="panel-state">
            Association data not available. Re-run the profiler at full depth to enable this tab.
          </div>

          <div v-else class="assoc-scroll">
            <AssociationTable
              v-model="secondaryColumn"
              :rows="assocRows"
            />
          </div>
        </div>

        <!-- Pair detail - shown once secondary column is selected -->
        <template v-if="secondaryColumn">

          <div class="pair-divider">
            <span class="pair-label">{{ primaryColumn }}</span>
            <span class="pair-sep">↔</span>
            <span class="pair-label">{{ secondaryColumn }}</span>
          </div>

          <PairStatsStrip
            :colA="primaryColumn"
            :colB="secondaryColumn"
            :intentA="primaryIntent"
            :intentB="secondaryIntent"
            :metric="selectedPair?.metric"
            :metricType="selectedPair?.metric_type"
            :strength="selectedPair?.strength"
          />

          <PairWarningsCard
            :colA="primaryColumn"
            :colB="secondaryColumn"
            :tasks="tasks"
          />

          <PairPlotCard
            :runKey="run.run_key"
            :colA="primaryColumn"
            :colB="secondaryColumn"
            :intentA="primaryIntent"
            :intentB="secondaryIntent"
            :theme="theme"
          />

        </template>

        <div v-else class="select-prompt">
          Click a row in the Associations panel to explore the pair in detail.
        </div>

      </div>

      <!-- Empty state: no column selected -->
      <div v-else class="empty-state card">
        <span class="empty-icon">↔</span>
        <p>Select a column from the browser to explore its associations.</p>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { getColumnAssociations } from '../../api.js'

import TooltipIcon              from '../../components/TooltipIcon.vue'
import ColumnBrowser            from '../../components/distributions/ColumnBrowser.vue'
import RelationshipsSummaryCard from '../../components/relationships/RelationshipsSummaryCard.vue'
import CorrelationHeatmapCard   from '../../components/relationships/CorrelationHeatmapCard.vue'
import AssociationTable         from '../../components/relationships/AssociationTable.vue'
import PairStatsStrip           from '../../components/relationships/PairStatsStrip.vue'
import PairWarningsCard         from '../../components/relationships/PairWarningsCard.vue'
import PairPlotCard             from '../../components/relationships/PairPlotCard.vue'

const props = defineProps({
  run:     { type: Object, required: true },
  tasks:   { type: Object, default: () => ({}) },
  figures: { type: Array,  default: () => [] },
  theme:   { type: String, default: 'dark' },
})

const primaryColumn    = ref(null)
const secondaryColumn  = ref(null)
const assocRows        = ref([])
const assocLoading     = ref(false)
const assocError       = ref(null)
const assocUnavailable = ref(false)

// Active filter from summary card warning chips (null | 'collinearity' | 'leakage')
const activeWarningFilter = ref(null)

// ── Column metadata ───────────────────────────────────────────────────────────

const primaryIntent = computed(() =>
  props.tasks.infer_types?.data?.[primaryColumn.value]?.analysis_intent_dtype ?? 'unknown'
)

const secondaryIntent = computed(() => {
  if (!secondaryColumn.value) return null
  const row = assocRows.value.find(r => r.column === secondaryColumn.value)
  return row?.col_other_intent ?? 'unknown'
})

const selectedPair = computed(() =>
  assocRows.value.find(r => r.column === secondaryColumn.value) ?? null
)

// Columns flagged for relationship-relevant issues
const relationshipWarnings = computed(() => {
  const vif        = props.tasks.detect_collinear_features?.data?.collinear_columns ?? []
  const leakage    = props.tasks.detect_data_leakage?.data?.leakage_pairs ?? {}
  const leakageCols = Object.keys(leakage).flatMap(k => k.split('|'))
  return new Set([...vif, ...leakageCols])
})

// ── Data loading ──────────────────────────────────────────────────────────────

async function loadAssociations(column) {
  assocLoading.value     = true
  assocError.value       = null
  assocUnavailable.value = false
  assocRows.value        = []
  secondaryColumn.value  = null

  try {
    const result = await getColumnAssociations(props.run.run_key, column)
    if (result.unavailable) {
      assocUnavailable.value = true
    } else {
      assocRows.value = result.associations.map(row => ({
        ...row,
        col_other_intent: row.col_b_intent === primaryIntent.value
          ? row.col_a_intent
          : row.col_b_intent,
      }))
    }
  } catch (e) {
    assocError.value = `Failed to load associations: ${e.message}`
  } finally {
    assocLoading.value = false
  }
}

function selectPrimary(col) {
  if (col === primaryColumn.value) return
  primaryColumn.value = col
}

watch(primaryColumn, col => { if (col) loadAssociations(col) })
</script>

<style scoped>
.relationships-tab {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* ── Two-pane exploration area ───────────────────────────────────────────── */
.tab-main {
  display: flex;
  gap: 16px;
  align-items: flex-start;
  min-height: 400px;
}

/* ── Browser pane ─────────────────────────────────────────────────────────── */
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

/* ── Column heading ──────────────────────────────────────────────────────── */
.col-heading {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  color: #f1f5f9;
  flex-wrap: wrap;
}
.col-heading-name  { font-family: monospace; font-size: 18px; }
.col-heading-badge {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.5px; padding: 3px 9px; border-radius: 9px; border: 1px solid;
}
.intent-continuous  { background: #1e3a5f; color: #60a5fa; border-color: #60a5fa; }
.intent-categorical { background: #14291f; color: #4ade80; border-color: #4ade80; }
.intent-boolean     { background: #2d1b4e; color: #c084fc; border-color: #c084fc; }
.intent-datetime    { background: #3d2510; color: #fb923c; border-color: #fb923c; }
.intent-text        { background: #3d0f29; color: #f472b6; border-color: #f472b6; }
.intent-unknown     { background: #1e293b; color: #94a3b8; border-color: #64748b; }

/* ── Associations card ───────────────────────────────────────────────────── */
.assoc-panel { padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; overflow: visible; }

.card-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.card-title {
  font-size: 13px;
  font-weight: 600;
  color: #e2e8f0;
  display: flex;
  align-items: center;
  gap: 4px;
}
.pair-count { font-size: 12px; color: #64748b; }

.assoc-scroll {
  max-height: 340px;
  overflow-y: auto;
  box-shadow: inset 0 -12px 12px -12px rgba(0,0,0,0.4);
  border-radius: 4px;
}

.panel-state { color: #64748b; font-size: 13px; padding: 24px 0; text-align: center; }
.error-text  { color: #f87171; }

/* ── Pair divider ────────────────────────────────────────────────────────── */
.pair-divider {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-top: 4px;
  border-top: 1px solid #334155;
}
.pair-label { font-size: 16px; font-weight: 700; color: #93c5fd; font-family: monospace; }
.pair-sep   { font-size: 16px; color: #64748b; }

/* ── Misc ────────────────────────────────────────────────────────────────── */
.select-prompt {
  color: #94a3b8;
  font-size: 13px;
  padding: 16px 0;
  text-align: center;
}

.empty-state {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 12px; color: #94a3b8; min-height: 300px;
}
.empty-icon { font-size: 32px; }
.empty-state p { font-size: 14px; text-align: center; margin: 0; }
</style>
