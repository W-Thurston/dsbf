<!-- dsbf/dashboard/frontend/src/components/distributions/TransformationPreviewCard.vue

  Shows transformation options for skewed continuous columns from
  transformation_preview. Each transform shows before/after skewness
  and key stats so users can decide whether to apply it.

  Only rendered for columns that appear in transformation_preview data.

  Props
  ─────
  column : String  - selected column name
  tasks  : Object  - pre-loaded task results
-->

<template>
  <div class="card tx-card">
    <div class="card-title">
      Transformation Preview
      <TooltipIcon
        text="Shows how common transformations would change the skewness of this column. Lower absolute skewness after transformation indicates a more symmetric distribution. Purely informational - no data is modified."
        align="left"
        direction="down"
      />
    </div>

    <!-- Not run -->
    <div v-if="state === 'not_run'" class="es-not-run">
      Transformation preview did not run for this column.
    </div>

    <!-- Ready -->
    <template v-else>

      <!-- Before stats strip -->
      <div class="tx-before-strip">
        <div class="tx-before-label">Original</div>
        <div class="tx-before-stats">
          <span class="tx-stat">
            <span class="tx-stat-label">Skewness</span>
            <span class="tx-stat-value" :class="skewClass(colData.original_skewness)">
              {{ fmtN(colData.original_skewness) }}
            </span>
          </span>
          <span class="tx-stat">
            <span class="tx-stat-label">Mean</span>
            <span class="tx-stat-value">{{ fmtN(colData.before_stats?.mean) }}</span>
          </span>
          <span class="tx-stat">
            <span class="tx-stat-label">Std Dev</span>
            <span class="tx-stat-value">{{ fmtN(colData.before_stats?.std) }}</span>
          </span>
          <span class="tx-stat">
            <span class="tx-stat-label">p5</span>
            <span class="tx-stat-value">{{ fmtN(colData.before_stats?.p5) }}</span>
          </span>
          <span class="tx-stat">
            <span class="tx-stat-label">p95</span>
            <span class="tx-stat-value">{{ fmtN(colData.before_stats?.p95) }}</span>
          </span>
        </div>
      </div>

      <!-- Transform options -->
      <div class="tx-list">
        <div
          v-for="tx in transforms"
          :key="tx.key"
          class="tx-item"
          :class="{
            'tx-item--recommended': tx.recommended,
            'tx-item--skipped':     tx.skipped,
          }"
        >
          <div class="tx-item-header">
            <span class="tx-name">{{ tx.display }}</span>
            <span v-if="tx.recommended"  class="tx-badge-recommended">recommended</span>
            <span v-else-if="tx.skipped" class="tx-badge-skipped">not applicable</span>
          </div>

          <div class="tx-suitable-for">{{ tx.suitable_for }}</div>

          <div v-if="tx.skipped" class="tx-skip-reason">{{ tx.skip_reason }}</div>

          <template v-else-if="tx.after_stats">
            <div class="tx-skew-row">
              <div class="tx-skew-stat">
                <span class="tx-skew-stat-label">Unmodified skewness</span>
                <span class="tx-skew-stat-val tx-skew-val--before">{{ fmtN(colData.original_skewness) }}</span>
              </div>
              <span class="tx-arrow">→</span>
              <div class="tx-skew-stat">
                <span class="tx-skew-stat-label">After transformation</span>
                <span
                  class="tx-skew-stat-val"
                  :class="Math.abs(tx.after_stats.skewness ?? 0) < Math.abs(colData.original_skewness)
                    ? 'tx-skew-val--better' : 'tx-skew-val--worse'"
                >{{ fmtN(tx.after_stats.skewness) }}</span>
              </div>
              <span
                v-if="tx.skew_reduction_pct != null"
                class="tx-reduction"
                :class="tx.skew_reduction_pct > 0 ? 'tx-reduction--good' : 'tx-reduction--bad'"
              >
                {{ tx.skew_reduction_pct > 0 ? '↓' : '↑' }}{{ Math.abs(tx.skew_reduction_pct).toFixed(0) }}%
              </span>
            </div>
          </template>
        </div>
      </div>

      <div class="tx-footer">Purely informational - no data is modified by DSBF.</div>

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

function fmtN(v) {
  if (v == null) return '-'
  const n = Number(v)
  if (!isFinite(n)) return '-'
  if (Math.abs(n) >= 10000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) >= 10)    return n.toFixed(2)
  return n.toPrecision(4).replace(/\.?0+$/, '')
}

function skewClass(v) {
  if (v == null) return ''
  return Math.abs(v) > 2 ? 'tx-skew--high' : Math.abs(v) > 1 ? 'tx-skew--mid' : 'tx-skew--low'
}

const colData = computed(() =>
  props.tasks?.transformation_preview?.data?.[props.column] ?? null
)

const state = computed(() => colData.value ? 'ready' : 'not_run')

const transforms = computed(() => {
  if (!colData.value?.transforms) return []
  const origSkew = Math.abs(colData.value?.original_skewness ?? 0)
  return Object.entries(colData.value.transforms).map(([key, tx]) => {
    const afterSkew = tx.after_stats?.skewness != null ? Math.abs(tx.after_stats.skewness) : null
    // Only recommend if the transform actually reduces absolute skewness
    const actuallyBetter = afterSkew != null && afterSkew < origSkew
    return {
      key,
      display:             tx.display      ?? key,
      suitable_for:        tx.suitable_for ?? '',
      skipped:             tx.skipped      ?? false,
      skip_reason:         tx.skip_reason  ?? '',
      after_stats:         tx.after_stats  ?? null,
      skew_reduction_pct:  tx.skew_reduction_pct ?? null,
      recommended:         (tx.recommended ?? false) && actuallyBetter,
    }
  })
})
</script>

<style scoped>
.tx-card { display: flex; flex-direction: column; gap: 14px; }
.es-not-run { color: #64748b; font-size: 13px; padding: 8px 0; text-align: center; }

.tx-before-strip {
  display: flex; align-items: center; gap: 16px;
  background: #0f172a; border: 1px solid #1e293b;
  border-radius: 6px; padding: 10px 14px; flex-wrap: wrap;
}
.tx-before-label {
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.5px; color: #64748b; flex-shrink: 0;
}
.tx-before-stats { display: flex; gap: 20px; flex-wrap: wrap; flex: 1; }
.tx-stat { display: flex; flex-direction: column; gap: 2px; align-items: center; }
.tx-stat-label {
  font-size: 10px; color: #64748b; text-transform: uppercase;
  letter-spacing: 0.4px; white-space: nowrap;
}
.tx-stat-value {
  font-size: 13px; font-weight: 600;
  font-family: ui-monospace, monospace; color: #e2e8f0;
}
.tx-skew--high { color: #f87171 !important; }
.tx-skew--mid  { color: #fb923c !important; }
.tx-skew--low  { color: #4ade80 !important; }

.tx-list { display: flex; flex-direction: column; gap: 8px; }

.tx-item {
  border: 1px solid #1e293b; border-radius: 6px;
  padding: 10px 14px; display: flex; flex-direction: column;
  gap: 6px; transition: border-color 0.15s;
}
.tx-item--recommended { border-color: #4ade80; background: rgba(74,222,128,0.03); }
.tx-item--skipped     { opacity: 0.5; }

.tx-item-header { display: flex; align-items: center; gap: 8px; }
.tx-name {
  font-size: 13px; font-weight: 600; color: #e2e8f0;
  font-family: ui-monospace, monospace;
}
.tx-badge-recommended {
  font-size: 10px; padding: 1px 7px; border-radius: 8px;
  background: #0f2718; color: #4ade80; border: 1px solid #4ade80;
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px;
}
.tx-badge-skipped {
  font-size: 10px; padding: 1px 7px; border-radius: 8px;
  background: #1e293b; color: #64748b; border: 1px solid #334155;
}

.tx-suitable-for { font-size: 11px; color: #64748b; }
.tx-skip-reason  { font-size: 11px; color: #64748b; font-style: italic; }

.tx-skew-row {
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
}
.tx-skew-stat {
  display: flex; flex-direction: column; gap: 2px; align-items: center;
}
.tx-skew-stat-label {
  font-size: 10px; color: #64748b; text-transform: uppercase;
  letter-spacing: 0.4px; white-space: nowrap;
}
.tx-skew-stat-val {
  font-size: 14px; font-weight: 700;
  font-family: ui-monospace, monospace;
}
.tx-skew-val--before { color: #94a3b8; }
.tx-skew-val--better { color: #4ade80; }
.tx-skew-val--worse  { color: #f87171; }

.tx-arrow { font-size: 12px; color: #64748b; }

.tx-reduction { font-size: 11px; font-weight: 700; margin-left: 4px; }
.tx-reduction--good { color: #4ade80; }
.tx-reduction--bad  { color: #f87171; }

.tx-footer { font-size: 11px; color: #334155; font-style: italic; text-align: right; }

:global(.theme-light) .tx-before-strip   { background: #f8fafc; border-color: #e2e8f0; }
:global(.theme-light) .tx-before-label   { color: #94a3b8; }
:global(.theme-light) .tx-stat-value     { color: #1e293b; }
:global(.theme-light) .tx-item           { border-color: #e2e8f0; }
:global(.theme-light) .tx-item--recommended { background: #f0fdf4; border-color: #86efac; }
:global(.theme-light) .tx-name           { color: #1e293b; }
:global(.theme-light) .tx-footer         { color: #94a3b8; }
</style>
