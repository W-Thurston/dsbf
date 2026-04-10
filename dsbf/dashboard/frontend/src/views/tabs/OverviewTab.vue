<template>
  <div class="overview-tab">

    <!-- Run metadata strip -->
    <div class="meta-strip card">
      <div class="meta-metric" v-for="m in runMetrics" :key="m.label">
        <span class="meta-label">
          {{ m.label }}
          <TooltipIcon :text="m.tooltip" align="center" />
        </span>
        <span class="meta-value"
          :title="m.fullValue ?? m.value"
          :data-key="m.dataKey"
          :class="{
            'adequacy-adequate':     m.dataKey === 'adequacy' && m.value === 'Adequate',
            'adequacy-limited':      m.dataKey === 'adequacy' && m.value === 'Limited',
            'adequacy-insufficient': m.dataKey === 'adequacy' && m.value === 'Insufficient',
          }"
        >{{ m.value }}</span>
      </div>
    </div>

    <!-- Row: data sample | run date + lineage stacked -->
    <div class="row-sample">
      <DataSampleTable :run-key="runKey" :run="run" />
      <div class="sample-sidebar">
        <RunDateCard :run="run" />
        <DataLineageCard :run="run" />
      </div>
    </div>

    <!-- Row: metadata table + alerts -->
    <div class="row-split">
      <ColumnMetadataTable :tasks="tasks" />
      <AlertsPanel :tasks="tasks" />
    </div>

    <!-- Row: type distribution + missingness matrix -->
    <div class="row-equal">
      <PlotCard
        title="Column Types Visualization"
        :figure="figureFor('dtype_stacked_bar', 'interactive')"
        :theme="theme"
      />
      <PlotCard
        title="Missingness Matrix"
        :figure="figureFor('missingness_matrix', 'static')"
        :theme="theme"
      />
    </div>

    <!-- Dtype optimization suggestions -->
    <DtypeOptimizationsCard :tasks="tasks" />

  </div>
</template>

<script setup>
import DtypeOptimizationsCard  from '../../components/overview/DtypeOptimizationsCard.vue'
import ColumnMetadataTable  from '../../components/overview/ColumnMetadataTable.vue'
import AlertsPanel          from '../../components/overview/AlertsPanel.vue'
import PlotCard             from '../../components/overview/PlotCard.vue'
import DataLineageCard      from '../../components/overview/DataLineageCard.vue'
import RunDateCard          from '../../components/overview/RunDateCard.vue'
import DataSampleTable      from '../../components/overview/DataSampleTable.vue'
import TooltipIcon          from '../../components/TooltipIcon.vue'
import { computed }         from 'vue'
import { figureFor as findFigure } from '../../utils.js'

const props = defineProps({
  runKey:  { type: String, default: null },
  run:     { type: Object, default: null },
  tasks:   { type: Object, required: true },
  figures: { type: Array,  default: () => [] },
  theme:   { type: String, default: 'dark' },
})

// figureFor is a thin wrapper over the shared util so templates stay readable
function figureFor(plotType, format) {
  return findFigure(props.figures, plotType, format, props.theme)
}

const runMetrics = computed(() => {
  if (!props.run) return []
  const shape    = props.tasks?.summarize_dataset_shape?.data ?? {}
  const dupCount = props.tasks?.detect_duplicates?.data?.duplicate_count ?? null
  const rowCount = props.run.row_count ?? null

  let dupValue = '-'
  if (dupCount != null) {
    const pct = rowCount ? ` (${((dupCount / rowCount) * 100).toFixed(1)}%)` : ''
    dupValue  = `${dupCount.toLocaleString()}${pct}`
  }

  return [
    {
      label:   'Rows',
      tooltip: 'Total number of rows (observations) in the dataset.',
      value:   rowCount?.toLocaleString() ?? '-',
    },
    {
      label:   'Columns',
      tooltip: 'Total number of columns (features) in the dataset.',
      value:   props.run.col_count?.toLocaleString() ?? '-',
    },
    {
      label:   'Memory',
      tooltip: 'Approximate memory footprint of the dataset when loaded into a pandas DataFrame.',
      value:   shape.approx_memory_MB != null ? `${shape.approx_memory_MB} MB` : '-',
    },
    {
      label:   'Missing',
      tooltip: 'Percentage of all cells in the dataset that contain a null or missing value.',
      value:   shape.null_cell_percentage != null
        ? `${(shape.null_cell_percentage * 100).toFixed(1)}%` : '-',
    },
    {
      label:   'Duplicate Rows',
      tooltip: 'Number of rows that are exact duplicates of another row, with their percentage of the total.',
      value:   dupValue,
      dataKey: 'dup',
    },
    {
      label:   'Sample Adequacy',
      tooltip: 'Whether the dataset has sufficient rows for reliable modelling. Based on rule-of-thumb checks for linear, tree-based, and general ML models.',
      value:   (() => {
        const verdict = props.tasks?.sample_size_adequacy?.data?.overall_verdict ?? null
        if (!verdict) return '-'
        return { adequate: 'Adequate', limited: 'Limited', insufficient: 'Insufficient' }[verdict] ?? verdict
      })(),
      dataKey: 'adequacy',
    },
    {
      label:   'Depth',
      tooltip: 'Profiling depth used for this run: basic (fast, core stats), standard (recommended), or full (all tasks including expensive checks).',
      value:   props.run.profiling_depth ?? '-',
    },
    {
      label:   'Stage',
      tooltip: 'Inferred lifecycle stage of the dataset - e.g. raw, exploratory, or modelling-ready.',
      value:   props.run.inferred_stage ?? '-',
    },
  ]
})
</script>

<style scoped>
.overview-tab { display: flex; flex-direction: column; gap: 16px; }

/* ── Meta strip ──────────────────────────────────────────────────────────── */
.meta-strip {
  display: flex;
  justify-content: space-around;
  flex-wrap: wrap;
  padding: 16px 24px;
}
.meta-metric {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  flex: 1;
  padding: 8px 16px;
  border-right: 1px solid #334155;
  min-width: 0;
}
.meta-metric:last-child { border-right: none; }
.meta-label {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  text-align: center;
  display: flex;
  align-items: center;
  gap: 2px;
}
.meta-value {
  font-size: 18px;
  font-weight: 600;
  color: #f1f5f9;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
  cursor: default;
}
.meta-metric:has(.meta-value[data-key="dup"]) .meta-value {
  font-size: 15px;
}
.meta-metric:has(.meta-value[data-key="adequacy"]) .meta-value[data-key="adequacy"] {
  /* coloured at render time via inline class - see template */
  font-size: 14px;
}

/* metadata table (wider) + alerts (narrower) - fixed height so both cards match */
.row-split {
  display: flex;
  gap: 16px;
  align-items: stretch;
  height: 500px;
}
.row-split > :first-child { flex: 2 1 0; min-width: 0; }
.row-split > :last-child  { flex: 1 1 0; min-width: 240px; }

/* Sample table + sidebar (run date stacked above lineage) */
.row-sample {
  display: flex;
  gap: 16px;
  align-items: stretch;
  /* No fixed height - sidebar drives the row height naturally,
     and the sample card stretches to match via height: 100% */
}

.row-sample > :first-child {
  flex: 3 1 0;
  min-width: 0;
  /* Let the card fill whatever height the sidebar sets */
  display: flex;
  flex-direction: column;
}

.sample-sidebar {
  flex: 1 1 0;
  min-width: 200px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* Lineage card fills remaining sidebar height after RunDateCard */
.sample-sidebar > :last-child {
  flex: 1 1 0;
}

/* equal-width chart pairs - fixed height so static and interactive cards match */
.row-equal { display: flex; gap: 16px; align-items: stretch; height: 420px; }
.row-equal > * { flex: 1 1 0; min-width: 0; }

/* Mid-breakpoint: stack row-sample before the sidebar hits min-width 200px
   and squeezes the sample table into an awkward intermediate state */
@media (max-width: 1100px) {
  .row-sample {
    flex-direction: column;
  }
  .row-sample > :first-child,
  .sample-sidebar {
    flex: none;
    width: 100%;
    min-width: 0;
    height: auto;
  }
  /* Sidebar cards lay out horizontally when there's full width available */
  .sample-sidebar {
    flex-direction: row;
    flex-wrap: wrap;
  }
  .sample-sidebar > * {
    flex: 1 1 240px;
    min-width: 0;
  }
}

@media (max-width: 900px) {
  /* Stack all multi-column rows vertically */
  .row-split,
  .row-sample,
  .row-equal {
    flex-direction: column;
    height: auto;       /* release fixed heights so each card takes natural height */
  }

  /* All children go full-width */
  .row-split > :first-child,
  .row-split > :last-child,
  .row-sample > :first-child,
  .sample-sidebar,
  .row-equal > * {
    flex: none;
    width: 100%;
    min-width: 0;
    height: auto;
  }

  /* Restore metadata table and alerts to sensible heights when stacked */
  .row-split > :first-child { max-height: 480px; }
  .row-split > :last-child  { max-height: 360px; }

  /* Each plot card gets its own natural height when stacked */
  .row-equal > * { min-height: 360px; }
}

/* ── Sample adequacy colouring ───────────────────────────────────────────── */
.adequacy-adequate     { color: #4ade80 !important; }
.adequacy-limited      { color: #fbbf24 !important; }
.adequacy-insufficient { color: #f87171 !important; }

/* ── Light theme overrides ───────────────────────────────────────────────── */
:global(body.theme-light) .meta-metric { border-right-color: #e2e8f0; }
:global(body.theme-light) .meta-label  { color: #94a3b8; }
:global(body.theme-light) .meta-value  { color: #1e293b; }
</style>
