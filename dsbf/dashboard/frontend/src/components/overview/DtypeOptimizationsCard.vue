<!-- dsbf/dashboard/frontend/src/components/overview/DtypeOptimizationsCard.vue

  Collapsible card showing dtype downcast suggestions from suggest_dtype_optimizations.
  Each row shows: column | current dtype → suggested dtype | estimated saving.

  Collapsed by default when all savings are minor; expanded when total
  potential saving exceeds 1 MB.

  Props
  ─────
  tasks : Object  - full tasks dict from RunDetailView
-->

<template>
  <div class="card dtype-card">

    <!-- Header - always visible, click to toggle -->
    <div class="dtype-header" @click="open = !open">
      <div class="dtype-header-left">
        <span class="dtype-title-wrap">
          <span class="card-title">Dtype Optimization Suggestions</span>
          <TooltipIcon
            text="Memory savings available by downcasting column dtypes - e.g. int64 → int8, object → category. Suggestions are purely advisory; DSBF never modifies your data."
            align="left"
            direction="down"
          />
        </span>
        <span class="dtype-summary" :class="summaryClass">
          <template v-if="taskState === 'ready'">
            {{ suggestions.length }} column{{ suggestions.length === 1 ? '' : 's' }} ·
            {{ totalSavingDisplay }} potential saving
          </template>
          <template v-else-if="taskState === 'empty'">All dtypes optimal</template>
          <template v-else-if="taskState === 'error'">Analysis error</template>
          <template v-else>Not available</template>
        </span>
      </div>
      <div class="dtype-header-right">
        <span class="dtype-note">Purely advisory - no data is modified</span>
        <button class="dtype-toggle-btn">{{ open ? '▲' : '▼' }}</button>
      </div>
    </div>

    <!-- Body - collapsible -->
    <Transition name="dtype-expand">
      <div v-if="open" class="dtype-body">
        <!-- State: task didn't run -->
        <div v-if="taskState === 'not_run'" class="es-not-run">
          Dtype optimization analysis did not run for this profiling depth.
        </div>

        <!-- State: task errored -->
        <div v-else-if="taskState === 'error'" class="es-error">
          <span>⚠</span>
          {{ errorMessage }}
        </div>

        <!-- State: ran but no suggestions -->
        <div v-else-if="taskState === 'empty'" class="es-empty">
          ✓ All column dtypes are already optimal - no downcast suggestions.
        </div>

        <!-- State: ready -->
        <div v-else class="dtype-table">
          <!-- Header -->
          <div class="dtype-row dtype-row--header">
            <span class="dtype-col dtype-col--name">Column</span>
            <span class="dtype-col dtype-col--current">Current</span>
            <span class="dtype-col dtype-col--arrow"></span>
            <span class="dtype-col dtype-col--suggested">Suggested</span>
            <span class="dtype-col dtype-col--reason">Reason</span>
            <span class="dtype-col dtype-col--saving">Saving</span>
          </div>
          <!-- Suggestion rows - scrollable after 10 rows -->
          <div class="dtype-scroll">
            <div
              v-for="s in suggestions"
              :key="s.column"
              class="dtype-row"
            >
              <span class="dtype-col dtype-col--name dtype-col-name">{{ s.column }}</span>
              <span class="dtype-col dtype-col--current dtype-current">{{ s.current_dtype }}</span>
              <span class="dtype-col dtype-col--arrow dtype-arrow">→</span>
              <span class="dtype-col dtype-col--suggested dtype-suggested">{{ s.suggested_dtype }}</span>
              <span class="dtype-col dtype-col--reason dtype-reason">{{ s.reason }}</span>
              <span class="dtype-col dtype-col--saving dtype-saving">{{ formatSaving(s.saving_bytes) }}</span>
            </div>
          </div>
        </div>
      </div>
    </Transition>

  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import TooltipIcon from '../TooltipIcon.vue'

const props = defineProps({
  tasks: { type: Object, default: () => ({}) },
})

const MB = 1024 * 1024
const KB = 1024

// data.suggestions is the column-keyed dict
const taskData = computed(
  () => props.tasks?.suggest_dtype_optimizations?.data?.suggestions ?? {}
)

const suggestions = computed(() => {
  const out = []
  for (const [col, info] of Object.entries(taskData.value)) {
    if (!info || typeof info !== 'object') continue
    if (!info.suggested_dtype) continue
    out.push({
      column:          col,
      current_dtype:   info.current_dtype   ?? '-',
      suggested_dtype: info.suggested_dtype,
      reason:          info.savings_note    ?? '',
      saving_bytes:    info.estimated_savings_bytes ?? 0,
    })
  }
  return out.sort((a, b) => b.saving_bytes - a.saving_bytes)
})

const taskResult = computed(() => props.tasks?.suggest_dtype_optimizations ?? null)

const taskState = computed(() => {
  if (!taskResult.value)                              return 'not_run'
  if (taskResult.value.status === 'error' ||
      taskResult.value.status === 'failure')          return 'error'
  if (suggestions.value.length === 0)                 return 'empty'
  return 'ready'
})

const errorMessage = computed(() =>
  taskResult.value?.error_metadata?.message ??
  taskResult.value?.summary?.message ??
  'This task encountered an error.'
)

const totalSavingBytes = computed(() =>
  suggestions.value.reduce((sum, s) => sum + (s.saving_bytes ?? 0), 0)
)

const totalSavingDisplay = computed(() => formatSaving(totalSavingBytes.value))

// Default closed; auto-open once data loads and saving is substantial
const open = ref(false)
watch(totalSavingBytes, (bytes) => {
  if (bytes >= MB) open.value = true
}, { immediate: true })

const summaryClass = computed(() => {
  if (totalSavingBytes.value >= 10 * MB) return 'dtype-summary--high'
  if (totalSavingBytes.value >= MB)      return 'dtype-summary--mid'
  return 'dtype-summary--low'
})

function formatSaving(bytes) {
  if (!bytes || bytes <= 0) return '-'
  if (bytes >= MB) return `${(bytes / MB).toFixed(1)} MB`
  if (bytes >= KB) return `${(bytes / KB).toFixed(0)} KB`
  return `${bytes} B`
}
</script>

<style scoped>
.dtype-card { padding: 0; overflow: visible; }

/* ── Header ──────────────────────────────────────────────────────────────── */
.dtype-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  cursor: pointer;
  user-select: none;
  gap: 16px;
  flex-wrap: wrap;
  transition: background 0.12s;
}
.dtype-header:hover { background: rgba(255,255,255,0.03); }

.dtype-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.dtype-header-left .card-title {
  margin-bottom: 0;
}
.dtype-title-wrap {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
/* Match the tooltip icon color to the table-header tooltips */
.dtype-title-wrap :deep(.tip-icon) { color: #64748b; }

.dtype-header-right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.dtype-summary {
  font-size: 12px;
  font-weight: 500;
}
.dtype-summary--high { color: #4ade80; }
.dtype-summary--mid  { color: #fbbf24; }
.dtype-summary--low  { color: #94a3b8; }

.dtype-note {
  font-size: 11px;
  color: #64748b;
  font-style: italic;
}

.dtype-toggle-btn {
  padding: 3px 8px;
  font-size: 11px;
  background: none;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #64748b;
  cursor: pointer;
  transition: all 0.12s;
}
.dtype-toggle-btn:hover { border-color: #60a5fa; color: #93c5fd; }

/* ── Body ────────────────────────────────────────────────────────────────── */
.dtype-body {
  border-top: 1px solid #1e293b;
  padding: 0 20px 16px;
}

/* ── Table ───────────────────────────────────────────────────────────────── */
.dtype-table {
  margin-top: 12px;
  width: 100%;
}

.dtype-row {
  display: grid;
  grid-template-columns: 2fr 1fr 20px 1fr 3fr 80px;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid #0f172a;
  font-size: 13px;
}
.dtype-scroll {
  max-height: 380px; /* ~10 rows at ~38px each */
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: #334155 transparent;
}
.dtype-scroll::-webkit-scrollbar { width: 5px; }
.dtype-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
.dtype-scroll .dtype-row:last-child { border-bottom: none; }
.dtype-row:hover:not(.dtype-row--header) { background: rgba(255,255,255,0.02); }

.dtype-row--header {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
  border-bottom: 1px solid #334155;
  padding-bottom: 8px;
  cursor: default;
}

.dtype-col { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.dtype-col-name {
  font-family: ui-monospace, 'Cascadia Code', monospace;
  font-size: 12px;
  color: #93c5fd;
}
.dtype-current   { color: #94a3b8; font-family: monospace; font-size: 12px; }
.dtype-arrow     { color: #64748b; text-align: center; flex-shrink: 0; }
.dtype-suggested { color: #4ade80; font-family: monospace; font-size: 12px; font-weight: 600; }
.dtype-reason    { color: #64748b; font-size: 12px; }
.dtype-saving    { color: #fbbf24; font-size: 12px; font-weight: 600; text-align: right; }

/* ── Empty states ────────────────────────────────────────────────────────── */
.es-not-run {
  color: #94a3b8;
  font-size: 13px;
  padding: 24px 0;
  text-align: center;
}
.es-error {
  color: #f87171;
  font-size: 13px;
  padding: 12px 16px;
  background: #3d0f0f;
  border-radius: 6px;
  border-left: 3px solid #f87171;
  margin: 8px 0;
  display: flex;
  align-items: flex-start;
  gap: 8px;
}
.es-empty {
  color: #4ade80;
  font-size: 13px;
  padding: 24px 0;
  text-align: center;
}

/* ── Expand/collapse transition ──────────────────────────────────────────── */
.dtype-expand-enter-active,
.dtype-expand-leave-active {
  transition: opacity 0.18s, max-height 0.22s ease;
  max-height: 1000px;
  overflow: hidden;
}
.dtype-expand-enter-from,
.dtype-expand-leave-to {
  opacity: 0;
  max-height: 0;
}

/* ── Light theme ─────────────────────────────────────────────────────────── */
:global(.theme-light) .dtype-header:hover  { background: rgba(0,0,0,0.02); }
:global(.theme-light) .dtype-body          { border-top-color: #e2e8f0; }
:global(.theme-light) .dtype-row           { border-bottom-color: #f1f5f9; }
:global(.theme-light) .dtype-row--header   { border-bottom-color: #e2e8f0; color: #94a3b8; }
:global(.theme-light) .dtype-row:hover:not(.dtype-row--header) { background: rgba(0,0,0,0.02); }
:global(.theme-light) .dtype-col-name      { color: #2563eb; }
:global(.theme-light) .dtype-current       { color: #64748b; }
:global(.theme-light) .dtype-reason        { color: #94a3b8; }
:global(.theme-light) .dtype-toggle-btn    { background: #f1f5f9; border-color: #cbd5e1; }
</style>
