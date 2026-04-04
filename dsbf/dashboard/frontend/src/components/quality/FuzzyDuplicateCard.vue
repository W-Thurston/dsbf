<!-- dsbf/dashboard/frontend/src/components/quality/FuzzyDuplicateCard.vue

  Shows near-duplicate row pairs from fuzzy_duplicate_detection.
  Rendered after the Redundancy dimension section in QualityTab.

  Each pair is a row-level finding stored under __dataset__ in the task data.
  Shows: row indices, similarity score, and the columns that matched.

  States:
  - not_run   : task absent
  - sampled   : task ran on a sample (not full dataset) - shown with reliability note
  - empty     : task ran, no fuzzy duplicates found
  - ready     : pairs found
-->

<template>
  <div class="card fd-card">
    <div class="fd-header" @click="open = !open">
      <div class="fd-header-left">
        <span class="fd-title-wrap">
          <span class="fd-status-dot" :class="statusDotClass" />
          <span class="card-title">Fuzzy Duplicate Detection</span>
        </span>
        <span v-if="state === 'ready' || state === 'sampled'" class="fd-subtitle">
          {{ pairCount }} near-duplicate pair{{ pairCount === 1 ? '' : 's' }} found
          <span v-if="state === 'sampled'" class="fd-sampled-note">(sampled dataset)</span>
        </span>
      </div>
      <button class="fd-toggle-btn">{{ open ? '▲' : '▼' }}</button>
    </div>

    <Transition name="fd-expand">
      <div v-if="open" class="fd-body">

        <!-- Not run -->
        <div v-if="state === 'not_run'" class="es-not-run">
          Fuzzy duplicate detection did not run for this profiling depth.
        </div>

        <!-- Error -->
        <div v-else-if="state === 'error'" class="es-error">
          <span>⚠</span> {{ errorMessage }}
        </div>

        <!-- Empty - positive -->
        <div v-else-if="state === 'empty'" class="es-empty">
          ✓ No near-duplicate rows detected at threshold {{ threshold }}.
        </div>

        <!-- Ready / Sampled -->
        <template v-else>

          <!-- Sampling reliability warning -->
          <div v-if="state === 'sampled'" class="fd-reliability-note">
            <span class="fd-rel-icon">⚠</span>
            Dataset exceeded the comparison limit - analysis ran on a
            {{ sampledRows?.toLocaleString() }} row sample.
            Additional near-duplicates may exist in the full dataset.
          </div>

          <!-- Threshold and method note -->
          <div class="fd-meta-row">
            <span class="fd-meta-item">
              Similarity threshold: <strong>{{ threshold }}</strong>
            </span>
            <span class="fd-meta-item">
              Method: token blocking + character similarity (SequenceMatcher)
            </span>
          </div>

          <!-- Pairs table -->
          <div class="fd-table">
            <div class="fd-row fd-row--header">
              <span class="fd-col fd-col--rows">Row pair</span>
              <span class="fd-col fd-col--score">Similarity</span>
              <span class="fd-col fd-col--cols">Differing columns</span>
            </div>
            <div class="fd-scroll">
              <div
                v-for="(pair, i) in visiblePairs"
                :key="i"
                class="fd-row"
              >
                <span class="fd-col fd-col--rows fd-row-ids">
                  {{ pair.row_i }} ↔ {{ pair.row_j }}
                </span>
                <span class="fd-col fd-col--score">
                  <span class="fd-score-bar-wrap">
                    <span
                      class="fd-score-bar"
                      :style="{ width: `${pair.similarity * 100}%` }"
                    />
                  </span>
                  <span class="fd-score-val">{{ (pair.similarity * 100).toFixed(1) }}%</span>
                </span>
                <span class="fd-col fd-col--cols fd-match-cols">
                  {{ pair.differing_columns?.join(', ') ?? '-' }}
                </span>
              </div>
            </div>
            <div v-if="pairCount > maxVisible" class="fd-truncated">
              Showing {{ maxVisible }} of {{ pairCount }} pairs.
              All pairs are stored in the task data output.
            </div>
          </div>

        </template>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  tasks: { type: Object, default: () => ({}) },
})

const open       = ref(false)
const maxVisible = 50

// ── State ─────────────────────────────────────────────────────────────────────

const taskResult = computed(() => props.tasks?.fuzzy_duplicate_detection ?? null)
const taskStatus = computed(() => taskResult.value?.status ?? null)

// Pairs live under __dataset__ key
const datasetEntry = computed(() =>
  taskResult.value?.data?.['__dataset__'] ?? null
)

const pairs = computed(() =>
  datasetEntry.value?.fuzzy_pairs ?? []
)

const pairCount   = computed(() => pairs.value.length)
const visiblePairs = computed(() => pairs.value.slice(0, maxVisible))

const threshold   = computed(() =>
  taskResult.value?.metadata?.similarity_threshold ??
  taskResult.value?.summary?.threshold ?? 0.85
)
const sampledRows = computed(() =>
  taskResult.value?.metadata?.sampled_rows ?? null
)

const state = computed(() => {
  if (!taskResult.value)                  return 'not_run'
  if (taskStatus.value === 'error' ||
      taskStatus.value === 'failure')     return 'error'
  if (pairCount.value === 0)              return 'empty'
  // Check if a reliability warning about sampling was emitted
  const hasReliabilityWarn = !!taskResult.value?.reliability_warnings?.heuristic_caution?.sampled
  if (hasReliabilityWarn || sampledRows.value != null) return 'sampled'
  return 'ready'
})

const errorMessage = computed(() =>
  taskResult.value?.error_metadata?.message ??
  taskResult.value?.summary?.message ??
  'This task encountered an error.'
)

const statusDotClass = computed(() => {
  if (state.value === 'empty' || state.value === 'not_run') return 'fd-dot--green'
  if (state.value === 'error') return 'fd-dot--amber'
  return 'fd-dot--amber'   // ready / sampled = pairs found = worth noting
})
</script>

<style scoped>
.fd-card { padding: 0; overflow: visible; }

/* ── Header ──────────────────────────────────────────────────────────────── */
.fd-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  cursor: pointer;
  user-select: none;
  transition: background 0.12s;
  gap: 12px;
}
.fd-header:hover { background: rgba(255,255,255,0.03); }

.fd-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.fd-title-wrap {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.fd-title-wrap .card-title { margin-bottom: 0; }
.fd-status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.fd-dot--green { background: #4ade80; box-shadow: 0 0 5px #4ade8055; }
.fd-dot--amber { background: #fbbf24; box-shadow: 0 0 5px #fbbf2455; }

.fd-subtitle      { font-size: 12px; color: #64748b; }
.fd-sampled-note  { color: #fbbf24; font-size: 11px; margin-left: 4px; }

.fd-toggle-btn {
  padding: 3px 8px;
  font-size: 11px;
  background: none;
  border: 1px solid #334155;
  border-radius: 4px;
  color: #64748b;
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.12s;
}
.fd-toggle-btn:hover { border-color: #60a5fa; color: #93c5fd; }

/* ── Body ────────────────────────────────────────────────────────────────── */
.fd-body {
  border-top: 1px solid #1e293b;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* ── Empty states ────────────────────────────────────────────────────────── */
.es-not-run { color: #64748b; font-size: 13px; padding: 16px 0; text-align: center; }
.es-empty   { color: #4ade80; font-size: 13px; padding: 16px 0; text-align: center; }
.es-error   { color: #f87171; font-size: 13px; padding: 12px; background: #3d0f0f; border-radius: 6px; border-left: 3px solid #f87171; display: flex; gap: 8px; }

/* ── Reliability note ────────────────────────────────────────────────────── */
.fd-reliability-note {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  background: #3d2a00;
  border: 1px solid #fbbf24;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 12px;
  color: #fde68a;
  line-height: 1.5;
}
.fd-rel-icon { flex-shrink: 0; }

/* ── Meta row ────────────────────────────────────────────────────────────── */
.fd-meta-row {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
}
.fd-meta-item { font-size: 12px; color: #64748b; }
.fd-meta-item strong { color: #e2e8f0; }

/* ── Table ───────────────────────────────────────────────────────────────── */
.fd-table { width: 100%; }

.fd-row {
  display: grid;
  grid-template-columns: 120px 180px 1fr;
  align-items: center;
  gap: 16px;
  padding: 7px 0;
  border-bottom: 1px solid #0f172a;
  font-size: 13px;
}
.fd-row:last-child { border-bottom: none; }
.fd-row:hover:not(.fd-row--header) { background: rgba(255,255,255,0.02); }

.fd-row--header {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748b;
  border-bottom: 1px solid #334155;
  padding-bottom: 8px;
}

.fd-col { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.fd-row-ids {
  font-family: ui-monospace, 'Cascadia Code', monospace;
  font-size: 12px;
  color: #93c5fd;
}

.fd-score-bar-wrap {
  display: inline-block;
  width: 80px;
  height: 6px;
  background: #1e293b;
  border-radius: 3px;
  overflow: hidden;
  vertical-align: middle;
  margin-right: 8px;
}
.fd-score-bar {
  display: block;
  height: 100%;
  background: #fbbf24;
  border-radius: 3px;
  transition: width 0.3s;
}
.fd-score-val {
  font-size: 12px;
  font-weight: 600;
  color: #fbbf24;
}

.fd-match-cols {
  font-size: 11px;
  color: #64748b;
}

.fd-scroll {
  max-height: 360px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: #334155 transparent;
}
.fd-scroll::-webkit-scrollbar { width: 5px; }
.fd-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

.fd-truncated {
  font-size: 11px;
  color: #64748b;
  padding: 10px 0 2px;
  text-align: center;
}

/* ── Transition ──────────────────────────────────────────────────────────── */
.fd-expand-enter-active,
.fd-expand-leave-active {
  transition: opacity 0.18s, max-height 0.22s ease;
  max-height: 1200px;
  overflow: hidden;
}
.fd-expand-enter-from,
.fd-expand-leave-to { opacity: 0; max-height: 0; }

/* ── Light theme ─────────────────────────────────────────────────────────── */
:global(.theme-light) .fd-header:hover    { background: rgba(0,0,0,0.02); }
:global(.theme-light) .fd-body            { border-top-color: #e2e8f0; }
:global(.theme-light) .fd-row             { border-bottom-color: #f1f5f9; }
:global(.theme-light) .fd-row--header     { border-bottom-color: #e2e8f0; color: #94a3b8; }
:global(.theme-light) .fd-row:hover:not(.fd-row--header) { background: rgba(0,0,0,0.02); }
:global(.theme-light) .fd-row-ids         { color: #2563eb; }
:global(.theme-light) .fd-score-bar-wrap  { background: #e2e8f0; }
:global(.theme-light) .fd-match-cols      { color: #94a3b8; }
:global(.theme-light) .fd-meta-item       { color: #94a3b8; }
:global(.theme-light) .fd-meta-item strong { color: #1e293b; }
:global(.theme-light) .fd-reliability-note { background: #fffbeb; border-color: #fcd34d; color: #92400e; }
:global(.theme-light) .es-not-run         { color: #94a3b8; }
</style>
