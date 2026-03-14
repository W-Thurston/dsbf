<!-- dsbf/dashboard/frontend/src/components/DataHealthBar.vue

  Persistent data-health status bar displayed above the tab bar on every page.
  Shows five dimension indicators (Completeness, Validity, Usability,
  Redundancy, Leakage) each with:
    • a traffic-light dot  (green / amber / red)
    • affected column count
    • tooltip on hover explaining the dimension and what drove the level

  Props
  ─────
  runKey  : String  — current run key, used to fetch /dq-status
  activeTab : String  — current tab key; used to subtly highlight the
                        dimensions that are most relevant to this tab

  The component fetches its own data independently so it doesn't need to
  wait for the parent's full task payload.
-->

<template>
  <div class="dh-bar" :class="{ 'dh-unavailable': !available }">
    <!-- Loading skeleton -->
    <template v-if="loading">
      <div v-for="n in 5" :key="n" class="dh-item dh-skeleton">
        <span class="dh-dot dh-dot--skeleton" />
        <span class="dh-skeleton-text" />
      </div>
    </template>

    <!-- Unavailable (scorer didn't run) -->
    <div v-else-if="!available" class="dh-na">
      Data health indicators unavailable for this run.
    </div>

    <!-- Indicators -->
    <template v-else>
      <div
        v-for="dim in dimensions"
        :key="dim.key"
        class="dh-item"
        :class="[
          `dh-item--${dim.level}`,
          { 'dh-item--active-tab': tabRelevance[activeTab]?.includes(dim.key) }
        ]"
        @mouseenter="hovered = dim.key"
        @mouseleave="hovered = null"
      >
        <!-- Dot -->
        <span class="dh-dot" :class="`dh-dot--${dim.level}`" />

        <!-- Label + count -->
        <span class="dh-label">{{ dim.label }}</span>
        <span class="dh-count" :class="`dh-count--${dim.level}`">
          {{ dim.affectedCount === 0 ? 'All clear' : `${dim.affectedCount} col${dim.affectedCount === 1 ? '' : 's'}` }}
        </span>

        <!-- Tooltip -->
        <Transition name="dh-tooltip">
          <div v-if="hovered === dim.key" class="dh-tooltip">
            <div class="dh-tooltip-title">{{ dim.label }}</div>
            <div class="dh-tooltip-body">{{ dim.description }}</div>
            <div v-if="dim.affectedCount > 0" class="dh-tooltip-stat">
              {{ dim.affectedCount }} of {{ totalColumns }} columns affected
              ({{ (dim.pctAffected * 100).toFixed(1) }}%)
            </div>
            <div v-else class="dh-tooltip-stat dh-tooltip-stat--good">
              No issues detected
            </div>
          </div>
        </Transition>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { getDqStatus } from '../api.js'

const props = defineProps({
  runKey:    { type: String, required: true },
  activeTab: { type: String, default: '' },
})

// ── State ────────────────────────────────────────────────────────────────────

const loading      = ref(true)
const available    = ref(false)
const totalColumns = ref(0)
const categories   = ref({})
const hovered      = ref(null)

// ── Fetch ─────────────────────────────────────────────────────────────────────

async function fetchStatus(runKey) {
  loading.value   = true
  available.value = false
  try {
    const data = await getDqStatus(runKey)
    available.value    = data.available ?? false
    totalColumns.value = data.total_columns ?? 0
    categories.value   = data.categories ?? {}
  } catch {
    available.value = false
  } finally {
    loading.value = false
  }
}

onMounted(() => fetchStatus(props.runKey))
watch(() => props.runKey, key => { if (key) fetchStatus(key) })

// ── Dimension config ──────────────────────────────────────────────────────────

const DIMENSION_META = {
  completeness: {
    label: 'Completeness',
    description:
      'Columns where 5% or more of values are missing. ' +
      'High missingness can bias analysis and reduce model reliability.',
  },
  validity: {
    label: 'Validity',
    description:
      'Columns with out-of-bounds values, constant columns (zero information), ' +
      'or columns that are structurally empty (>95% zeros). These indicate ' +
      'data collection or pipeline problems.',
  },
  usability: {
    label: 'Usability',
    description:
      'Columns that are structurally unsuitable for direct analysis — likely ' +
      'ID columns, near-constant dominant values, or extremely high cardinality ' +
      'categoricals that would need transformation before use.',
  },
  redundancy: {
    label: 'Redundancy',
    description:
      'Numeric columns with high multicollinearity (VIF > 10). Redundant ' +
      'features carry duplicate information and can destabilise models.',
  },
  leakage: {
    label: 'Leakage',
    description:
      'Column pairs with near-perfect correlation that may indicate data ' +
      'leakage — one column encoding the same information as another. ' +
      'This is a correctness risk, not just a modelling inefficiency.',
  },
}

const dimensions = computed(() =>
  Object.entries(DIMENSION_META).map(([key, meta]) => {
    const cat = categories.value[key] ?? {}
    return {
      key,
      label:         meta.label,
      description:   meta.description,
      level:         cat.level         ?? 'green',
      affectedCount: cat.affected_count ?? 0,
      pctAffected:   cat.pct_affected   ?? 0,
    }
  })
)

// ── Tab relevance ─────────────────────────────────────────────────────────────
// Which dimensions to highlight (brighter opacity) based on the active tab.
// Dimensions not in the active tab's list are dimmed slightly — not hidden.

const tabRelevance = {
  overview:      ['completeness', 'validity', 'usability', 'redundancy', 'leakage'],
  distributions: ['completeness', 'validity', 'usability'],
  relationships: ['redundancy', 'leakage'],
  quality:       ['completeness', 'validity', 'usability', 'redundancy', 'leakage'],
}
</script>

<style scoped>
/* ── Bar container ─────────────────────────────────────────────────────────── */
.dh-bar {
  display: flex;
  align-items: stretch;
  background: var(--card-bg, #1e293b);
  border: 1px solid var(--border, #334155);
  border-radius: 10px;
  padding: 0;
  margin-bottom: 12px;
  overflow: visible;
  min-height: 56px;
}

.dh-unavailable {
  justify-content: center;
  align-items: center;
  padding: 12px 24px;
}

.dh-na {
  font-size: 13px;
  color: #475569;
}

/* ── Individual indicator ──────────────────────────────────────────────────── */
.dh-item {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3px;
  flex: 1;
  padding: 12px 8px;
  border-right: 1px solid var(--border, #334155);
  transition: background 0.15s, opacity 0.2s;
  cursor: default;
}

.dh-item:last-child { border-right: none; }

/* Dim dimensions not relevant to the current tab — but only if a tab that
   defines relevance is active. Overview and quality show all at full opacity. */
.dh-bar:has(.dh-item--active-tab) .dh-item:not(.dh-item--active-tab) {
  opacity: 0.4;
}

.dh-item:hover {
  background: var(--hover-bg, rgba(255,255,255,0.04));
  opacity: 1 !important;
}

/* ── Traffic-light dot ─────────────────────────────────────────────────────── */
.dh-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dh-dot--green   { background: #4ade80; box-shadow: 0 0 6px #4ade8066; }
.dh-dot--amber   { background: #fbbf24; box-shadow: 0 0 6px #fbbf2466; }
.dh-dot--red     { background: #f87171; box-shadow: 0 0 6px #f8717166; }
.dh-dot--skeleton { background: #334155; }

/* ── Label ─────────────────────────────────────────────────────────────────── */
.dh-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
  white-space: nowrap;
}

/* ── Count ─────────────────────────────────────────────────────────────────── */
.dh-count {
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.dh-count--green { color: #4ade80; }
.dh-count--amber { color: #fbbf24; }
.dh-count--red   { color: #f87171; }

/* ── Skeleton ──────────────────────────────────────────────────────────────── */
.dh-skeleton {
  pointer-events: none;
}
.dh-skeleton-text {
  display: block;
  width: 48px;
  height: 10px;
  background: #334155;
  border-radius: 4px;
  animation: dh-pulse 1.4s ease-in-out infinite;
}

@keyframes dh-pulse {
  0%, 100% { opacity: 0.5; }
  50%       { opacity: 1;   }
}

/* ── Tooltip ───────────────────────────────────────────────────────────────── */
.dh-tooltip {
  position: absolute;
  top: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
  width: 220px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 12px 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  pointer-events: none;
}

/* Keep tooltip inside viewport for the first and last items */
.dh-item:first-child .dh-tooltip {
  left: 0;
  transform: none;
}
.dh-item:last-child .dh-tooltip {
  left: auto;
  right: 0;
  transform: none;
}

.dh-tooltip-title {
  font-size: 12px;
  font-weight: 700;
  color: #e2e8f0;
  margin-bottom: 6px;
}

.dh-tooltip-body {
  font-size: 12px;
  color: #94a3b8;
  line-height: 1.5;
  margin-bottom: 8px;
}

.dh-tooltip-stat {
  font-size: 11px;
  font-weight: 600;
  color: #fbbf24;
  border-top: 1px solid #1e293b;
  padding-top: 6px;
}

.dh-tooltip-stat--good { color: #4ade80; }

/* ── Tooltip transition ────────────────────────────────────────────────────── */
.dh-tooltip-enter-active,
.dh-tooltip-leave-active { transition: opacity 0.12s, transform 0.12s; }
.dh-tooltip-enter-from,
.dh-tooltip-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(-4px);
}
.dh-item:first-child .dh-tooltip-enter-from,
.dh-item:first-child .dh-tooltip-leave-to {
  transform: translateY(-4px);
}
.dh-item:last-child .dh-tooltip-enter-from,
.dh-item:last-child .dh-tooltip-leave-to {
  transform: translateY(-4px);
}

/* ── Light theme overrides ─────────────────────────────────────────────────── */
:global(.theme-light) .dh-bar {
  background: #f8fafc;
  border-color: #e2e8f0;
}
:global(.theme-light) .dh-item {
  border-right-color: #e2e8f0;
}
:global(.theme-light) .dh-item:hover {
  background: rgba(0,0,0,0.03);
}
:global(.theme-light) .dh-label { color: #94a3b8; }
:global(.theme-light) .dh-tooltip {
  background: #ffffff;
  border-color: #e2e8f0;
  box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
:global(.theme-light) .dh-tooltip-title { color: #1e293b; }
:global(.theme-light) .dh-tooltip-body  { color: #64748b; }
:global(.theme-light) .dh-tooltip-stat  { border-top-color: #e2e8f0; }
</style>
