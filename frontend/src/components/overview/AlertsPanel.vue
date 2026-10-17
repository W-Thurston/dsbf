<template>
  <div class="card alerts-card">
    <div class="card-title">Data Signals</div>

    <!-- Filters -->
    <div class="filters">
      <div class="filter-group">
        <span class="filter-label">Priority</span>
        <div class="filter-pills">
          <button
            v-for="s in severities" :key="s.key"
            class="pill" :class="[`pill-${s.key}`, { active: activeSeverities.has(s.key) }]"
            @click="toggleSeverity(s.key)"
          >{{ s.label }}</button>
        </div>
      </div>
      <div class="filter-group">
        <span class="filter-label">Type</span>
        <div class="filter-pills">
          <button
            v-for="t in availableTypes" :key="t"
            class="pill pill-type" :class="{ active: activeTypes.has(t) }"
            @click="toggleType(t)"
          >{{ t }}</button>
        </div>
      </div>
    </div>

    <div v-if="noTasksLoaded" class="es-not-run">Loading signals…</div>

    <div v-else-if="filteredAlerts.length === 0" class="no-alerts">
      {{ alerts.length === 0 ? '✓ Nothing flagged.' : '✓ Nothing matches the current filters.' }}
    </div>

    <div v-else class="alerts-list">
      <div
        v-for="(alert, i) in filteredAlerts" :key="i"
        class="alert-item" :class="`alert-${alert.level}`"
      >
        <span class="alert-icon">{{ alert.icon }}</span>
        <span class="alert-msg">{{ alert.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  tasks: { type: Object, default: () => ({}) },
})

// ── Empty state ───────────────────────────────────────────────────────────────
const noTasksLoaded = computed(() => Object.keys(props.tasks).length === 0)

// ── Severity filter ──────────────────────────────────────────────────────────
const severities = [
  { key: 'error', label: '● Priority 1' },
  { key: 'warn',  label: '● Priority 2' },
  { key: 'info',  label: '● Priority 3' },
]
const activeSeverities = ref(new Set(['error', 'warn', 'info']))

function toggleSeverity(key) {
  const s = new Set(activeSeverities.value)
  s.has(key) ? s.delete(key) : s.add(key)
  activeSeverities.value = s
}

// ── Alert type filter ─────────────────────────────────────────────────────────
const activeTypes = ref(new Set())   // empty = all types shown

function toggleType(type) {
  const s = new Set(activeTypes.value)
  s.has(type) ? s.delete(type) : s.add(type)
  activeTypes.value = s
}

// ── Alert generation ─────────────────────────────────────────────────────────
const alerts = computed(() => {
  const out = []

  const highNull = props.tasks.summarize_nulls?.data?.high_null_columns ?? []
  for (const col of highNull) {
    const pct = props.tasks.summarize_nulls?.data?.null_percentages?.[col] ?? 0
    out.push({ level: pct >= 0.5 ? 'error' : 'warn', icon: '⚠️', type: 'Missingness',
      message: `${col} has ${(pct * 100).toFixed(1)}% missing values` })
  }

  const hcData    = props.tasks.detect_high_cardinality?.data    ?? {}
  const hcSummary = props.tasks.detect_high_cardinality?.summary ?? {}
  const threshold = hcSummary.threshold ?? 50
  for (const [col, count] of Object.entries(hcData)) {
    if (typeof count === 'number' && count > threshold) {
      out.push({ level: 'warn', icon: '🔢', type: 'Cardinality',
        message: `${col} has high cardinality (${count.toLocaleString()} unique values)` })
    }
  }

  const constCols = props.tasks.detect_constant_columns?.data?.constant_columns ?? []
  for (const col of constCols) {
    out.push({ level: 'error', icon: '🚫', type: 'Constant',
      message: `${col} appears constant - only one unique value found` })
  }

  const dupData   = props.tasks.detect_duplicate_columns?.data ?? {}
  const dupGroups = dupData.duplicate_groups ?? []
  for (const group of dupGroups) {
    out.push({ level: 'warn', icon: '♊', type: 'Duplicates',
      message: `Likely duplicate columns: ${group.join(', ')}` })
  }

  const skewData     = props.tasks.detect_skewness?.data ?? {}
  const highlySkewed = Object.entries(skewData)
    .filter(([, v]) => typeof v === 'number' && Math.abs(v) > 2)
    .map(([col, v]) => ({ col, v }))
  for (const { col, v } of highlySkewed) {
    out.push({ level: 'info', icon: '📐', type: 'Skewness',
      message: `${col} is highly skewed (skew = ${v.toFixed(2)})` })
  }

  const domData = props.tasks.detect_single_dominant_value?.data ?? {}
  for (const [col, info] of Object.entries(domData)) {
    if (info?.dominance_level === 'high') {
      out.push({ level: 'warn', icon: '📊', type: 'Dominance',
        message: `${col} has a dominant value - "${info.mode}" covers ${(info.mode_proportion * 100).toFixed(1)}% of rows` })
    }
  }

  for (const [taskName, result] of Object.entries(props.tasks)) {
    const warnings = result?.reliability_warnings ?? {}
    for (const [, codes] of Object.entries(warnings)) {
      for (const [, info] of Object.entries(codes)) {
        if (info?.description) {
          out.push({ level: 'info', icon: '🔍', type: 'Reliability',
            message: `${taskName.replaceAll('_', ' ')}: ${info.description}` })
        }
      }
    }
  }

  return out
})

// ── Available type options derived from actual alerts ─────────────────────────
const availableTypes = computed(() => [...new Set(alerts.value.map(a => a.type))].sort())

// ── Filtered output ───────────────────────────────────────────────────────────
const filteredAlerts = computed(() => {
  return alerts.value.filter(a => {
    const severityOk = activeSeverities.value.has(a.level)
    const typeOk     = activeTypes.value.size === 0 || activeTypes.value.has(a.type)
    return severityOk && typeOk
  })
})
</script>

<style scoped>
.alerts-card { display: flex; flex-direction: column; height: 100%; }

/* ── Filters ─────────────────────────────────────────────────────────────── */
.filters {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid #1e293b;
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.filter-label {
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  min-width: 52px;
  flex-shrink: 0;
}

.filter-pills { display: flex; gap: 4px; flex-wrap: wrap; }

.pill {
  padding: 2px 9px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid transparent;
  background: #0f172a;
  color: #64748b;
  transition: all 0.15s;
}

.pill-error.active { background: #3d0f0f; color: #f87171; border-color: #f87171; }
.pill-warn.active  { background: #3d2510; color: #fb923c; border-color: #fb923c; }
.pill-info.active  { background: #1e3a5f; color: #60a5fa; border-color: #60a5fa; }
.pill-type.active  { background: #14291f; color: #4ade80; border-color: #4ade80; }
.pill:not(.active):hover { background: #1e293b; color: #94a3b8; }

/* ── Alerts list ─────────────────────────────────────────────────────────── */
.no-alerts   { color: #4ade80; font-size: 13px; padding: 8px 0; }
.es-not-run  { color: #64748b; font-size: 13px; padding: 24px 0; text-align: center; }
.alerts-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  overflow-y: auto;
  flex: 1;
  min-height: 0;
}

.alert-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
  border-left: 3px solid;
}
.alert-error { background: #3d0f0f; border-color: #f87171; color: #fca5a5; }
.alert-warn  { background: #3d2510; border-color: #fb923c; color: #fed7aa; }
.alert-info  { background: #1e3a5f; border-color: #60a5fa; color: #bfdbfe; }

.alert-icon { flex-shrink: 0; }
.alert-msg  { line-height: 1.4; }
</style>
