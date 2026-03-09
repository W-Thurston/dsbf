<template>
  <div class="overview-tab">

    <!-- Row: data sample | run date + lineage stacked -->
    <div class="row-sample">
      <DataSampleTable :run-key="run?.run_key" :run="run" />
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

    <!-- Correlation matrix (full width) -->
    <PlotCard
      title="Correlation Matrix"
      :figure="figureFor('correlation_matrix', 'interactive')"
      :theme="theme"
    />


  </div>
</template>

<script setup>
import ColumnMetadataTable  from '../../components/overview/ColumnMetadataTable.vue'
import AlertsPanel          from '../../components/overview/AlertsPanel.vue'
import PlotCard             from '../../components/overview/PlotCard.vue'
import DataLineageCard      from '../../components/overview/DataLineageCard.vue'
import RunDateCard          from '../../components/overview/RunDateCard.vue'
import DataSampleTable      from '../../components/overview/DataSampleTable.vue'
import { figureFor as findFigure } from '../../utils.js'

const props = defineProps({
  run:     { type: Object, default: null },
  tasks:   { type: Object, required: true },
  figures: { type: Array,  default: () => [] },
  theme:   { type: String, default: 'dark' },
})

// figureFor is a thin wrapper over the shared util so templates stay readable
function figureFor(plotType, format) {
  return findFigure(props.figures, plotType, format, props.theme)
}
</script>

<style scoped>
.overview-tab { display: flex; flex-direction: column; gap: 16px; }

/* metadata table (wider) + alerts (narrower) */
.row-split {
  display: flex;
  gap: 16px;
  align-items: stretch;
}
.row-split > :first-child { flex: 2 1 0; min-width: 0; }
.row-split > :last-child  { flex: 1 1 0; min-width: 240px; }

/* Sample table + sidebar (run date stacked above lineage) */
.row-sample {
  display: flex;
  gap: 16px;
  align-items: stretch;
}

.row-sample > :first-child {
  flex: 3 1 0;
  min-width: 0;
}

.sample-sidebar {
  flex: 1 1 0;
  min-width: 200px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* equal-width chart pairs */
.row-equal { display: flex; gap: 16px; align-items: stretch; }
.row-equal > * { flex: 1 1 0; min-width: 0; }

@media (max-width: 900px) {
  .row-split,
  .row-equal {
    flex-direction: column;
  }
  .row-split > :first-child,
  .row-split > :last-child,
  .row-sample > :first-child,
  .sample-sidebar,
  .row-equal > * {
    flex: none;
    width: 100%;
    min-width: 0;
  }
}
</style>
