<!-- frontend/src/components/charts/DsbfChart.vue

  Universal chart wrapper for all DSBF ECharts visualizations.

  Handles:
  - Theme switching (dark / light) in sync with the dashboard theme prop
  - Responsive resizing via ResizeObserver
  - PNG and SVG export with one click
  - Loading and empty states with consistent styling
  - Optional column selector tab strip (for multi-column charts)

  Usage:
    <DsbfChart
      :option="myEChartsOption"
      title="ACF / PACF"
      :theme="theme"
      height="260px"
    />

  Props
  ─────
  option       Object    ECharts option object (reactive — updates chart on change)
  title        String    Card header title
  subtitle     String    Optional subtitle shown below the title
  theme        String    "dark" | "light"  (from RunDetailView)
  height       String    CSS height of the chart canvas  (default: "300px")
  loading      Boolean   Show skeleton loader
  empty        Boolean   Show empty state instead of chart
  emptyMessage String    Text shown in empty state
  columns      Array     If provided, renders a column selector tab strip
                         Each item: { key: String, label: String }
  modelValue   String    Currently selected column key (v-model)
  exportName   String    Base filename for downloads  (default: "dsbf-chart")

  Emits
  ─────
  update:modelValue   When the user selects a different column tab
-->

<template>
  <div class="dsbf-chart-card card">
    <!-- ── Header ──────────────────────────────────────────────────────────── -->
    <div class="dc-header">
      <div class="dc-title-wrap">
        <span class="dc-title">{{ title }}</span>
        <span v-if="subtitle" class="dc-subtitle">{{ subtitle }}</span>
      </div>
      <div class="dc-actions">
        <button
          v-if="!empty && !loading"
          class="dc-action-btn"
          title="Download high-resolution PNG (2x)"
          @click="exportPng"
        >
          ↓ 2x PNG
        </button>
      </div>
    </div>

    <!-- ── Column selector ─────────────────────────────────────────────────── -->
    <div v-if="columns && columns.length > 1" class="dc-col-tabs">
      <button
        v-for="col in columns"
        :key="col.key"
        class="dc-col-tab"
        :class="{ 'dc-col-tab--active': modelValue === col.key }"
        @click="$emit('update:modelValue', col.key)"
      >
        {{ col.label ?? col.key }}
      </button>
    </div>

    <!-- ── Chart canvas / states ────────────────────────────────────────────── -->
    <div
      class="dc-body"
      :class="{ 'dc-body--fill': fillParent }"
      :style="fillParent ? {} : { height }"
    >
      <!-- Loading skeleton -->
      <div v-if="loading" class="dc-skeleton">
        <div
          v-for="n in 5"
          :key="n"
          class="dc-skeleton-bar"
          :style="{ height: `${[60, 85, 40, 70, 50][n - 1]}%`, animationDelay: `${n * 0.1}s` }"
        />
      </div>

      <!-- Empty state -->
      <div v-else-if="empty" class="dc-empty">
        <span class="dc-empty-icon">◫</span>
        <span class="dc-empty-msg">{{ emptyMessage || 'No data available.' }}</span>
      </div>

      <!-- ECharts canvas -->
      <v-chart
        v-else
        ref="chartRef"
        class="dc-echart"
        :theme="echartsTheme"
        :option="option"
        :autoresize="true"
        :update-options="{ notMerge: false, lazyUpdate: true }"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import VChart from 'vue-echarts'
import type { EChartsOption } from 'echarts'
import { registerDsbfThemes } from '../../chartTheme'

// vue-echarts requires explicit component registration of the ECharts modules
// you use. Import only what's needed to keep the bundle small.
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import {
  BarChart,
  LineChart,
  HeatmapChart,
  ScatterChart,
  BoxplotChart,
  CustomChart,
} from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
  TitleComponent,
  MarkLineComponent,
  MarkAreaComponent,
} from 'echarts/components'

use([
  CanvasRenderer,
  BarChart,
  LineChart,
  HeatmapChart,
  ScatterChart,
  BoxplotChart,
  CustomChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
  TitleComponent,
  MarkLineComponent,
  MarkAreaComponent,
])

// Registered here, not in main.ts, so echarts loads only with the first chart.
registerDsbfThemes()

// ── Props & emits ─────────────────────────────────────────────────────────────
export interface ChartColumn {
  key: string
  label?: string
}

const props = withDefaults(
  defineProps<{
    option: EChartsOption
    title?: string
    subtitle?: string
    theme?: 'dark' | 'light'
    height?: string
    fillParent?: boolean
    loading?: boolean
    empty?: boolean
    emptyMessage?: string
    columns?: ChartColumn[] | null
    modelValue?: string | null
    exportName?: string
  }>(),
  {
    title: '',
    subtitle: '',
    theme: 'dark',
    height: '300px',
    fillParent: false,
    loading: false,
    empty: false,
    emptyMessage: '',
    columns: null,
    modelValue: null,
    exportName: 'dsbf-chart',
  },
)

defineEmits<{ 'update:modelValue': [key: string] }>()

// ── Internals ─────────────────────────────────────────────────────────────────
const chartRef = ref<InstanceType<typeof VChart> | null>(null)
const echartsTheme = computed(() => (props.theme === 'light' ? 'dsbf-light' : 'dsbf-dark'))

// ── Export helpers ────────────────────────────────────────────────────────────
function exportPng() {
  const chart = chartRef.value?.chart
  if (!chart) return
  const url = chart.getDataURL({
    type: 'png',
    pixelRatio: 2, // retina-quality for slides
    backgroundColor: props.theme === 'light' ? '#ffffff' : '#0f172a',
    excludeComponents: ['toolbox'],
  })
  download(url, `${props.exportName}.png`)
}

function download(url: string, filename: string) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
}
</script>

<style scoped>
.dsbf-chart-card {
  display: flex;
  flex-direction: column;
  gap: 0;
  padding: 0;
  overflow: hidden;
}

/* ── Header ────────────────────────────────────────────────────────────────── */
.dc-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px 10px;
  gap: 12px;
}

.dc-title-wrap {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.dc-title {
  font-size: 13px;
  font-weight: 600;
  color: #e2e8f0;
}

.dc-subtitle {
  font-size: 11px;
  color: #64748b;
}

.dc-actions {
  display: flex;
  gap: 6px;
  flex-shrink: 0;
}

.dc-action-btn {
  padding: 3px 10px;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #64748b;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.12s;
  white-space: nowrap;
}
.dc-action-btn:hover {
  border-color: #60a5fa;
  color: #93c5fd;
}

/* ── Column tabs ───────────────────────────────────────────────────────────── */
.dc-col-tabs {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  padding: 0 18px 10px;
}

.dc-col-tab {
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
.dc-col-tab:hover {
  border-color: #60a5fa;
  color: #93c5fd;
}
.dc-col-tab--active {
  background: #0f2035;
  border-color: #60a5fa;
  color: #93c5fd;
}

/* ── Body / canvas area ────────────────────────────────────────────────────── */
.dc-body {
  position: relative;
  width: 100%;
  min-height: 80px;
}
/* When used inside a flex container with a fixed height, fill remaining space */
.dc-body--fill {
  flex: 1;
  min-height: 0;
}

.dc-echart {
  width: 100%;
  height: 100%;
}

/* ── Loading skeleton ─────────────────────────────────────────────────────── */
.dc-skeleton {
  display: flex;
  align-items: flex-end;
  gap: 6px;
  padding: 16px 18px;
  height: 100%;
}

.dc-skeleton-bar {
  flex: 1;
  border-radius: 3px;
  background: linear-gradient(90deg, #1e293b 25%, #334155 50%, #1e293b 75%);
  background-size: 200% 100%;
  animation: dc-shimmer 1.4s infinite;
}

@keyframes dc-shimmer {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}

/* ── Empty state ───────────────────────────────────────────────────────────── */
.dc-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 8px;
  color: #475569;
  padding: 24px;
}

.dc-empty-icon {
  font-size: 28px;
  opacity: 0.4;
}
.dc-empty-msg {
  font-size: 12px;
  text-align: center;
  line-height: 1.5;
}

/* ── Light theme ───────────────────────────────────────────────────────────── */
:global(.theme-light) .dc-title {
  color: #1e293b;
}
:global(.theme-light) .dc-action-btn {
  background: #f8fafc;
  border-color: #e2e8f0;
  color: #94a3b8;
}
:global(.theme-light) .dc-action-btn:hover {
  border-color: #2563eb;
  color: #2563eb;
}
:global(.theme-light) .dc-col-tab {
  background: #f1f5f9;
  border-color: #e2e8f0;
  color: #94a3b8;
}
:global(.theme-light) .dc-col-tab:hover {
  border-color: #2563eb;
  color: #2563eb;
}
:global(.theme-light) .dc-col-tab--active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1d4ed8;
}
:global(.theme-light) .dc-skeleton-bar {
  background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%);
  background-size: 200% 100%;
}
:global(.theme-light) .dc-empty {
  color: #94a3b8;
}
</style>
