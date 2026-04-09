<!-- dsbf/dashboard/frontend/src/components/relationships/CorrelationHeatmapCard.vue

  Placeholder for the Pearson correlation matrix heatmap.
  The full D3 visualization is deferred to the D3 refactor.
  The matrix data is available at:
    tasks.compute_pairwise_associations.data['__correlation_matrix__']

  Props
  ─────
  tasks : Object
-->
<template>
  <div class="card heatmap-card">

    <div class="heatmap-header">
      <span class="heatmap-title">
        Correlation Matrix
        <TooltipIcon
          text="Pearson correlation coefficients between all continuous columns. Cells near ±1 indicate strong linear relationships. Clicking a cell will select that column pair for detailed exploration below."
          direction="down"
          align="left"
        />
      </span>
      <span v-if="numericColCount" class="heatmap-meta">
        {{ numericColCount }} × {{ numericColCount }} · continuous columns only
      </span>
    </div>

    <div class="heatmap-placeholder">
      <span class="placeholder-icon">▦</span>
      <div class="placeholder-text">
        <strong>D3 heatmap visualization coming in the next refactor.</strong>
        <span v-if="numericColCount">
          A {{ numericColCount }}×{{ numericColCount }} Pearson r matrix has been computed and is ready to render.
        </span>
        <span v-else>
          Correlation matrix not available — re-run at standard depth or higher.
        </span>
      </div>
    </div>

  </div>
</template>

<script setup>
import { computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  tasks: { type: Object, default: () => ({}) },
})

const numericColCount = computed(() => {
  const matrix = props.tasks?.compute_pairwise_associations?.data?.['__correlation_matrix__']
  if (!matrix || typeof matrix !== 'object') return null
  return Object.keys(matrix).length
})
</script>

<style scoped>
.heatmap-card {
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow: visible;
}

.heatmap-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}

.heatmap-title {
  font-size: 13px;
  font-weight: 600;
  color: #e2e8f0;
  display: flex;
  align-items: center;
  gap: 4px;
}

.heatmap-meta {
  font-size: 12px;
  color: #64748b;
}

.heatmap-placeholder {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 24px 20px;
  background: #0f172a;
  border: 1px dashed #334155;
  border-radius: 8px;
  color: #64748b;
}

.placeholder-icon {
  font-size: 36px;
  opacity: 0.25;
  flex-shrink: 0;
}

.placeholder-text {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
  line-height: 1.6;
}

.placeholder-text strong {
  color: #94a3b8;
}
</style>
