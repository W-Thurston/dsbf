<template>
  <div class="pair-plot-card card">
    <div class="card-title-row">
      <span class="card-title">{{ colA }} <span class="pair-sep">↔</span> {{ colB }}</span>
      <span class="plot-type-badge">{{ plotTypeLabel }}</span>
    </div>

    <div v-if="loading" class="plot-loading">Loading data…</div>
    <div v-else-if="error" class="plot-error">{{ error }}</div>
    <div v-else-if="!hasData" class="plot-empty">
      Source data not available for this run. Ensure source_path is set.
    </div>
    <div v-else ref="plotEl" class="plot-area" />

    <div v-if="sampledRows && totalRows && sampledRows < totalRows" class="sample-note">
      Showing {{ sampledRows.toLocaleString() }} of {{ totalRows.toLocaleString() }} rows
    </div>
  </div>
</template>

<script setup>
import { ref, watch, computed, onMounted, nextTick } from 'vue'
import { getColumnData } from '../../api.js'

const props = defineProps({
  runKey:   { type: String, required: true },
  colA:     { type: String, required: true },
  colB:     { type: String, required: true },
  intentA:  { type: String, default: 'continuous' },
  intentB:  { type: String, default: 'continuous' },
  theme:    { type: String, default: 'dark' },
  maxRows:  { type: Number, default: 3000 },
})

const plotEl    = ref(null)
const loading   = ref(false)
const error     = ref(null)
const hasData   = ref(false)
const totalRows   = ref(null)
const sampledRows = ref(null)

// ── Plot type dispatch ────────────────────────────────────────────────────────
const plotTypeLabel = computed(() => {
  const a = props.intentA, b = props.intentB
  if (a === 'continuous' && b === 'continuous')   return 'Scatter + Regression'
  if (a === 'continuous' && b === 'categorical')  return 'Box Plot by Group'
  if (a === 'categorical' && b === 'continuous')  return 'Box Plot by Group'
  if (a === 'categorical' && b === 'categorical') return 'Grouped Bar Chart'
  return 'Plot'
})

// ── Theme colours ─────────────────────────────────────────────────────────────
const colours = computed(() => props.theme === 'dark'
  ? { bg: '#0f172a', paper: '#1e293b', text: '#e2e8f0', grid: '#334155',
      scatter: '#60a5fa', line: '#f97316', zero: '#475569' }
  : { bg: '#f8fafc', paper: '#ffffff', text: '#1e293b', grid: '#e2e8f0',
      scatter: '#2563eb', line: '#ea580c', zero: '#94a3b8' }
)

// ── Regression helpers ────────────────────────────────────────────────────────
function linreg(xs, ys) {
  const pairs = xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => x != null && y != null)
  const n = pairs.length
  if (n < 2) return null
  const sx = pairs.reduce((s, [x]) => s + x, 0)
  const sy = pairs.reduce((s, [, y]) => s + y, 0)
  const sxy = pairs.reduce((s, [x, y]) => s + x * y, 0)
  const sxx = pairs.reduce((s, [x]) => s + x * x, 0)
  const syy = pairs.reduce((s, [, y]) => s + y * y, 0)
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
  const intercept = (sy - slope * sx) / n
  const ssRes = pairs.reduce((s, [x, y]) => s + (y - (slope * x + intercept)) ** 2, 0)
  const ssTot = pairs.reduce((s, [, y]) => s + (y - sy / n) ** 2, 0)
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0
  const xmin = Math.min(...pairs.map(([x]) => x))
  const xmax = Math.max(...pairs.map(([x]) => x))
  return { slope, intercept, r2, xmin, xmax }
}

// ── Build Plotly traces ───────────────────────────────────────────────────────
function buildTraces(data) {
  const a = props.intentA, b = props.intentB
  const c = colours.value
  const valA = data[props.colA]
  const valB = data[props.colB]

  if (a === 'continuous' && b === 'continuous') {
    const reg = linreg(valA, valB)
    const traces = [{
      type: 'scatter', mode: 'markers',
      x: valA, y: valB,
      marker: { color: c.scatter, size: 5, opacity: 0.55 },
      name: `${props.colA} vs ${props.colB}`,
    }]
    if (reg) {
      const xs = [reg.xmin, reg.xmax]
      const ys = xs.map(x => reg.slope * x + reg.intercept)
      traces.push({
        type: 'scatter', mode: 'lines',
        x: xs, y: ys,
        line: { color: c.line, width: 2, dash: 'solid' },
        name: `Fit (r²=${reg.r2.toFixed(3)})`,
      })
    }
    return { traces, r2: reg?.r2, layout_extra: {
      xaxis: { title: props.colA },
      yaxis: { title: props.colB },
    }}
  }

  // continuous × categorical → grouped box
  const [contCol, catCol, contVals, catVals] =
    a === 'continuous'
      ? [props.colA, props.colB, valA, valB]
      : [props.colB, props.colA, valB, valA]

  if ((a === 'continuous' && b === 'categorical') || (a === 'categorical' && b === 'continuous')) {
    const groups = {}
    catVals.forEach((cat, i) => {
      if (cat == null || contVals[i] == null) return
      const key = String(cat)
      if (!groups[key]) groups[key] = []
      groups[key].push(contVals[i])
    })
    const PALETTE = ['#60a5fa','#4ade80','#fb923c','#f472b6','#a78bfa','#34d399','#fbbf24','#f87171']
    const traces = Object.entries(groups)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([cat, vals], i) => ({
        type: 'box', y: vals, name: cat,
        marker: { color: PALETTE[i % PALETTE.length] },
        boxpoints: vals.length < 100 ? 'outliers' : false,
      }))
    return { traces, layout_extra: {
      xaxis: { title: catCol },
      yaxis: { title: contCol },
    }}
  }

  // categorical × categorical → grouped bar (top 10 per axis)
  if (a === 'categorical' && b === 'categorical') {
    const counts = {}
    valA.forEach((va, i) => {
      if (va == null || valB[i] == null) return
      const ka = String(va), kb = String(valB[i])
      if (!counts[ka]) counts[ka] = {}
      counts[ka][kb] = (counts[ka][kb] || 0) + 1
    })
    const topA = Object.entries(counts)
      .sort((a, b) => Object.values(b[1]).reduce((s, v) => s + v, 0) - Object.values(a[1]).reduce((s, v) => s + v, 0))
      .slice(0, 10).map(([k]) => k)
    const allB = [...new Set(valB.filter(v => v != null).map(String))].slice(0, 10)
    const PALETTE = ['#60a5fa','#4ade80','#fb923c','#f472b6','#a78bfa','#34d399','#fbbf24','#f87171']
    const traces = allB.map((kb, i) => ({
      type: 'bar', name: kb,
      x: topA,
      y: topA.map(ka => counts[ka]?.[kb] || 0),
      marker: { color: PALETTE[i % PALETTE.length] },
    }))
    return { traces, layout_extra: {
      barmode: 'group',
      xaxis: { title: props.colA },
      yaxis: { title: 'Count' },
    }}
  }

  return { traces: [], layout_extra: {} }
}

// ── Render ────────────────────────────────────────────────────────────────────
async function render() {
  if (!props.colA || !props.colB || !props.runKey) return
  loading.value = true
  error.value   = null
  hasData.value = false

  try {
    const data = await getColumnData(props.runKey, [props.colA, props.colB], props.maxRows)
    if (data.error) { error.value = data.error; return }

    totalRows.value   = data.total_rows
    sampledRows.value = data.sampled_rows
    hasData.value     = true
    loading.value     = false

    await nextTick()
    if (!plotEl.value) return

    const Plotly = await import('plotly.js-dist-min')
    const c = colours.value
    const { traces, layout_extra, r2 } = buildTraces(data)

    const layout = {
      margin:  { t: 20, r: 20, b: 60, l: 60 },
      paper_bgcolor: c.paper,
      plot_bgcolor:  c.bg,
      font:    { color: c.text, size: 12 },
      showlegend: traces.length > 1,
      legend:  { bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
      xaxis: { gridcolor: c.grid, zerolinecolor: c.zero, ...layout_extra.xaxis },
      yaxis: { gridcolor: c.grid, zerolinecolor: c.zero, ...layout_extra.yaxis },
      ...layout_extra,
    }

    Plotly.react(plotEl.value, traces, layout, {
      responsive: true, displayModeBar: false,
    })
  } catch (e) {
    error.value = `Failed to load data: ${e.message}`
    loading.value = false
  }
}

onMounted(render)
watch(() => [props.colA, props.colB, props.runKey, props.theme], render)
</script>

<style scoped>
.pair-plot-card { padding: 16px 20px; }
.card-title-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.card-title { font-size: 15px; font-weight: 600; color: #e2e8f0; }
.pair-sep { color: #60a5fa; margin: 0 6px; }
.plot-type-badge {
  font-size: 11px; padding: 3px 10px; border-radius: 10px;
  background: #1e293b; color: #60a5fa; border: 1px solid #334155;
}
.plot-area { width: 100%; height: 360px; }
.plot-loading, .plot-empty, .plot-error {
  height: 360px; display: flex; align-items: center; justify-content: center;
  color: #475569; font-size: 13px;
}
.plot-error { color: #f87171; }
.sample-note { font-size: 11px; color: #475569; text-align: right; margin-top: 6px; }
</style>
