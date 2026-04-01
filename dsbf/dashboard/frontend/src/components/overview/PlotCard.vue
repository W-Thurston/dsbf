<template>
  <div class="card plot-card">
    <div class="card-title-row">
      <div class="card-title">{{ title }}</div>
      <slot name="controls" /></div>

    <div v-if="loading" class="loading">Loading chart…</div>
    <div v-else-if="error" class="plot-error">
      <span class="plot-error-icon">⚠</span>
      {{ error }}
    </div>

    <!-- Plotly interactive -->
    <div v-else-if="isInteractive && plotData" ref="plotEl" class="plot-container" />

    <!-- Static image - wrapped so it centers and fills the card height -->
    <div v-else-if="imageUrl" class="plot-img-wrap">
      <img :src="imageUrl" class="plot-img" :alt="title" />
    </div>

    <!-- No figure provided at all (task didn't produce this plot type) -->
    <div v-else-if="!props.figure" class="no-figure">
      <span class="no-figure-icon">📊</span>
      Chart not available for this run.
      <span class="no-figure-hint">Re-run at full depth to generate this visualisation.</span>
    </div>

    <!-- Figure provided but nothing rendered (theme mismatch or other) -->
    <div v-else class="no-figure">No figure available for this theme.</div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import Plotly from 'plotly.js-dist-min'
import { figureFileUrl } from '../../api.js'

const props = defineProps({
  title:   { type: String, required: true },
  figure:  { type: Object, default: null },
  theme:   { type: String, default: 'dark' },
})

const plotEl   = ref(null)
const loading  = ref(false)
const error    = ref(null)
const imageUrl = ref(null)
const plotData = ref(null)

const isInteractive = computed(() => props.figure?.format === 'interactive')

const darkLayout = {
  paper_bgcolor: '#1e293b',
  plot_bgcolor:  '#1e293b',
  font:          { color: '#94a3b8' },
}
const lightLayout = {
  paper_bgcolor: '#ffffff',
  plot_bgcolor:  '#f8fafc',
  font:          { color: '#334155' },
}

async function loadFigure() {
  error.value    = null
  imageUrl.value = null
  plotData.value = null

  if (!props.figure) return

  loading.value = true
  try {
    const url = figureFileUrl(props.figure.id)

    if (isInteractive.value) {
      const resp = await fetch(url)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const raw = await resp.json()
      // Handle common Panel/Plotly serialisation formats:
      //   A: { data: [...], layout: {} }           - standard plotly figure
      //   B: { figure: { data: [...], layout: {} }} - Panel-wrapped
      //   C: [ ...traces ]                          - bare data array
      if (Array.isArray(raw)) {
        plotData.value = { data: raw, layout: {} }
      } else if (raw.figure) {
        plotData.value = raw.figure
      } else if (raw.data !== undefined) {
        plotData.value = raw
      } else {
        console.warn('[PlotCard] Unrecognised JSON structure:', raw)
        throw new Error('Unexpected JSON structure. Keys: ' + Object.keys(raw).join(', '))
      }

      // Set loading=false BEFORE renderPlotly so the ref div is in the DOM
      loading.value = false
      await nextTick()
      renderPlotly()
      return  // skip the finally assignment below
    } else {
      imageUrl.value = url
    }
  } catch (e) {
    error.value = `Failed to load figure: ${e.message}`
  } finally {
    loading.value = false
  }
}

/**
 * Compute dynamic layout overrides for heatmap-type charts (correlation matrix).
 * Scales height, margins, and font sizes based on the number of columns so that
 * labels remain readable even with many features.
 */
function heatmapOverrides(data) {
  const trace = data.find(t => t.type === 'heatmap')
  if (!trace) return {}

  const n = Math.max(
    Array.isArray(trace.x) ? trace.x.length : 0,
    Array.isArray(trace.y) ? trace.y.length : 0,
  )
  if (n === 0) return {}

  // Each cell needs ~36px minimum; clamp between 360 and 900
  const cellPx  = Math.max(36, Math.min(64, Math.floor(720 / n)))
  const height   = Math.min(900, Math.max(360, n * cellPx))

  // Axis label font: starts at 13px, shrinks as n grows, minimum 8px
  const axisFontSize = Math.max(8, Math.round(13 - (n - 5) * 0.35))

  // Annotation font (cell text): hide below 10px - too small to read
  const annotFontSize = Math.max(7, Math.round(12 - (n - 5) * 0.4))
  const showAnnot     = annotFontSize >= 8

  // Margins: left/bottom need more room for long axis labels
  const maxLabelLen = Math.max(
    ...(trace.x ?? []).map(l => String(l).length),
    ...(trace.y ?? []).map(l => String(l).length),
  )
  const marginLB = Math.min(160, Math.max(60, maxLabelLen * axisFontSize * 0.65))

  const overrides = {
    height,
    margin: { l: marginLB, r: 20, t: 40, b: marginLB },
    xaxis: { tickfont: { size: axisFontSize }, tickangle: -45 },
    yaxis: { tickfont: { size: axisFontSize } },
  }

  // Scale down or hide cell annotations if present
  if (Array.isArray(data[0]?.text) || showAnnot === false) {
    overrides.annotations = (plotData.value.layout?.annotations ?? []).map(a => ({
      ...a,
      font: { ...a.font, size: showAnnot ? annotFontSize : 0 },
      visible: showAnnot,
    }))
  }

  return overrides
}

function renderPlotly() {
  if (!plotEl.value || !plotData.value) return
  const themeLayout = props.theme === 'dark' ? darkLayout : lightLayout
  const dynamic     = heatmapOverrides(plotData.value.data)

  const layout = {
    ...plotData.value.layout,
    ...themeLayout,
    autosize: true,
    margin: { l: 50, r: 20, t: 40, b: 50 },  // default; overridden by dynamic if heatmap
    ...dynamic,
    // theme colours override everything, re-apply after dynamic spread
    paper_bgcolor: themeLayout.paper_bgcolor,
    plot_bgcolor:  themeLayout.plot_bgcolor,
    font:          { ...themeLayout.font, ...(dynamic.font ?? {}) },
  }

  Plotly.react(plotEl.value, plotData.value.data, layout, {
    responsive: true,
    displayModeBar: false,
  })
}

// Re-render (don't re-fetch) when only theme changes
watch(() => props.theme, () => {
  if (isInteractive.value && plotData.value) renderPlotly()
})

// Re-fetch when the figure itself changes
watch(() => props.figure?.id, loadFigure)

onMounted(loadFigure)

onBeforeUnmount(() => {
  if (plotEl.value) Plotly.purge(plotEl.value)
})
</script>

<style scoped>
.plot-card { min-height: 200px; height: 100%; display: flex; flex-direction: column; box-sizing: border-box; }

.card-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  margin-bottom: 0;
}
.card-title-row .card-title { margin-bottom: 0; }

.card-title { flex-shrink: 0; }

.plot-container,
.plot-img,
.loading,
.no-figure,
.plot-error { flex: 1; }

.loading, .no-figure {
  color: #94a3b8;
  font-size: 13px;
  padding: 40px 0;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.no-figure-icon { font-size: 24px; opacity: 0.4; }
.no-figure-hint { font-size: 11px; color: #475569; }

.plot-error {
  color: #f87171;
  font-size: 13px;
  padding: 16px;
  text-align: center;
  background: #3d0f0f;
  border-radius: 6px;
  border-left: 3px solid #f87171;
  margin: 8px 0;
  display: flex;
  align-items: flex-start;
  gap: 8px;
  text-align: left;
}
.plot-error-icon { flex-shrink: 0; }

.plot-container {
  width: 100%;
  min-height: 360px;
  /* height is set dynamically by Plotly layout for heatmaps */
}

.plot-img-wrap {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  min-height: 0;
}

.plot-img {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 6px;
  display: block;
}
</style>
