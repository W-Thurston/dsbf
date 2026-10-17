<!-- dsbf/dashboard/frontend/src/components/distributions/QQPlotCard.vue

  Renders a Q-Q (quantile-quantile) plot for normality assessment using
  pre-computed quantile pairs from normality_qq_plots.

  The plot is drawn as an inline SVG - no D3 required. Points are the
  empirical vs theoretical quantiles; the reference line shows where
  a perfectly normal distribution would lie.

  Points close to the line = approximately normal.
  Deviation at the tails = heavy tails / skew.

  Props
  ─────
  column : String  - selected column name
  tasks  : Object  - pre-loaded task results
-->

<template>
  <div class="card qq-card">
    <div class="card-title">
      Normality Q-Q Plot
      <TooltipIcon
        text="Plots empirical quantiles against theoretical normal quantiles. Points close to the diagonal reference line indicate approximately normal data. Deviation at the tails reveals skewness or heavy-tailed behaviour."
        align="left"
        direction="down"
      />
    </div>

    <!-- Not run -->
    <div v-if="state === 'not_run'" class="es-not-run">
      Q-Q plot data not available - normality tests may not have run for this column.
    </div>

    <!-- Normal column excluded -->
    <div v-else-if="state === 'normal'" class="es-empty">
      ✓ This column passed normality tests - Q-Q plot not computed.
    </div>

    <!-- Plot -->
    <template v-else>
      <div class="qq-meta">
        <span class="qq-meta-item">n = {{ colData.n?.toLocaleString() }}</span>
        <span class="qq-meta-item" v-if="colData.mean_abs_deviation != null">
          Mean deviation from line: <strong>{{ colData.mean_abs_deviation.toFixed(4) }}</strong>
        </span>
        <span class="qq-normal-result" :class="normalClass">{{ normalLabel }}</span>
      </div>

      <div class="qq-plot-wrap" ref="wrapEl">
        <svg
          :viewBox="`0 0 ${W} ${H}`"
          class="qq-svg"
          xmlns="http://www.w3.org/2000/svg"
        >
          <!-- Axes -->
          <line :x1="PAD" :y1="PAD" :x2="PAD" :y2="H - PAD" class="qq-axis" />
          <line :x1="PAD" :y1="H - PAD" :x2="W - PAD" :y2="H - PAD" class="qq-axis" />

          <!-- Axis labels -->
          <text :x="W / 2" :y="H - 4" class="qq-axis-label" text-anchor="middle">Theoretical quantiles</text>
          <text
            :x="10"
            :y="H / 2"
            class="qq-axis-label"
            text-anchor="middle"
            :transform="`rotate(-90, 10, ${H / 2})`"
          >Sample quantiles</text>

          <!-- Reference line -->
          <line
            :x1="scaleX(refLine.x1)"
            :y1="scaleY(refLine.y1)"
            :x2="scaleX(refLine.x2)"
            :y2="scaleY(refLine.y2)"
            class="qq-ref-line"
          />

          <!-- Points -->
          <circle
            v-for="(pt, i) in points"
            :key="i"
            :cx="scaleX(pt.x)"
            :cy="scaleY(pt.y)"
            r="3"
            class="qq-point"
          />
        </svg>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  column: { type: String, required: true },
  tasks:  { type: Object, default: () => ({}) },
})

// SVG dimensions
const W   = 420
const H   = 320
const PAD = 44

// ── Data ─────────────────────────────────────────────────────────────────────

const colData = computed(() =>
  props.tasks?.normality_qq_plots?.data?.[props.column] ?? null
)

const normalityResult = computed(() =>
  props.tasks?.normality_tests?.data?.[props.column]?.result ?? null
)

const state = computed(() => {
  if (!props.tasks?.normality_qq_plots) return 'not_run'
  if (!colData.value) {
    // Task ran but no entry for this column - likely normal and excluded
    return normalityResult.value === 'normal' ? 'normal' : 'not_run'
  }
  return 'ready'
})

const points = computed(() => {
  const d = colData.value
  if (!d) return []
  const theoretical = d.theoretical ?? []
  const empirical   = d.empirical   ?? []
  return theoretical.map((x, i) => ({ x, y: empirical[i] ?? 0 }))
})


// ── Scales ────────────────────────────────────────────────────────────────────
// Compute point bounds independently so refLine can reference them without
// a circular dependency (refLine used to reference xMin which referenced refLine).

const pointsXMin = computed(() => points.value.length ? Math.min(...points.value.map(p => p.x)) : -3)
const pointsXMax = computed(() => points.value.length ? Math.max(...points.value.map(p => p.x)) :  3)
const pointsYMin = computed(() => points.value.length ? Math.min(...points.value.map(p => p.y)) : -3)
const pointsYMax = computed(() => points.value.length ? Math.max(...points.value.map(p => p.y)) :  3)

const refLine = computed(() => {
  const d = colData.value
  if (!d) return { x1: -3, y1: -3, x2: 3, y2: 3 }
  if (d.reference_line) {
    const rl = d.reference_line
    const x1 = rl.start_x ?? pointsXMin.value
    const x2 = rl.end_x   ?? pointsXMax.value
    return {
      x1,
      y1: rl.slope * x1 + rl.intercept,
      x2,
      y2: rl.slope * x2 + rl.intercept,
    }
  }
  // Fallback: diagonal through point extremes
  const pts = points.value
  if (!pts.length) return { x1: -3, y1: -3, x2: 3, y2: 3 }
  return { x1: pts[0].x, y1: pts[0].y, x2: pts[pts.length - 1].x, y2: pts[pts.length - 1].y }
})

const xMin = computed(() => Math.min(pointsXMin.value, refLine.value.x1))
const xMax = computed(() => Math.max(pointsXMax.value, refLine.value.x2))
const yMin = computed(() => Math.min(pointsYMin.value, refLine.value.y1))
const yMax = computed(() => Math.max(pointsYMax.value, refLine.value.y2))

const plotW = W - PAD * 2
const plotH = H - PAD * 2

function scaleX(v) {
  const range = xMax.value - xMin.value || 1
  return PAD + ((v - xMin.value) / range) * plotW
}
function scaleY(v) {
  const range = yMax.value - yMin.value || 1
  return H - PAD - ((v - yMin.value) / range) * plotH
}

// ── Normal label ──────────────────────────────────────────────────────────────

const normalLabel = computed(() => {
  const r = normalityResult.value
  if (r === 'normal')     return 'Consistent with normal'
  if (r === 'non_normal') return 'Non-normal'
  return 'Normality not tested'
})

const normalClass = computed(() => {
  const r = normalityResult.value
  if (r === 'normal')     return 'qq-normal--pass'
  if (r === 'non_normal') return 'qq-normal--fail'
  return 'qq-normal--unknown'
})
</script>

<style scoped>
.qq-card { display: flex; flex-direction: column; gap: 12px; }

.es-not-run { color: #64748b; font-size: 13px; padding: 8px 0; text-align: center; }
.es-empty   { color: #4ade80; font-size: 13px; padding: 8px 0; text-align: center; }

/* ── Meta row ────────────────────────────────────────────────────────────── */
.qq-meta {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  font-size: 12px;
}
.qq-meta-item { color: #64748b; }
.qq-meta-item strong { color: #94a3b8; }

.qq-normal-result {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 9px;
  border: 1px solid;
}
.qq-normal--pass    { background: #0f2718; color: #4ade80; border-color: #4ade80; }
.qq-normal--fail    { background: #3d2510; color: #fb923c; border-color: #fb923c; }
.qq-normal--unknown { background: #1e293b; color: #94a3b8; border-color: #64748b; }

/* ── SVG ─────────────────────────────────────────────────────────────────── */
.qq-plot-wrap {
  width: 100%;
  overflow: hidden;
}
.qq-svg {
  width: 100%;
  height: auto;
  display: block;
}

.qq-axis {
  stroke: #334155;
  stroke-width: 1;
}
.qq-axis-label {
  font-size: 10px;
  fill: #64748b;
  font-family: ui-sans-serif, system-ui, sans-serif;
}
.qq-ref-line {
  stroke: #60a5fa;
  stroke-width: 1.5;
  stroke-dasharray: 5 3;
  opacity: 0.7;
}
.qq-point {
  fill: #fbbf24;
  opacity: 0.65;
  transition: opacity 0.1s;
}
.qq-point:hover { opacity: 1; }

/* ── Light theme ─────────────────────────────────────────────────────────── */
:global(.theme-light) .qq-axis       { stroke: #e2e8f0; }
:global(.theme-light) .qq-axis-label { fill: #94a3b8; }
:global(.theme-light) .qq-ref-line   { stroke: #2563eb; }
:global(.theme-light) .qq-point      { fill: #d97706; }
</style>
