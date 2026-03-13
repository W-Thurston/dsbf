<template>
  <div class="card guidance-card">
    <div class="card-title">Guidance</div>

    <div v-if="!blurbs.length" class="no-guidance">
      No notable characteristics detected for this column.
    </div>

    <div v-else class="insights">
      <div
        v-for="(blurb, i) in blurbs"
        :key="i"
        class="insight-item"
        :class="[`insight-${blurb.level}`, { 'is-open': openSet.has(i) }]"
      >
        <!-- Clickable title row - always visible -->
        <button class="insight-trigger" @click="toggle(i)" :aria-expanded="openSet.has(i)">
          <span class="insight-icon">{{ levelIcon(blurb.level) }}</span>
          <span class="insight-title">{{ blurb.title }}</span>
          <span class="insight-chevron" :class="{ rotated: openSet.has(i) }">›</span>
        </button>

        <!-- Expandable body + actions -->
        <Transition name="blurb-slide">
          <div v-if="openSet.has(i)" class="insight-body-wrap">
            <p class="insight-body">{{ blurb.body }}</p>
            <div v-if="blurb.actions?.length" class="insight-actions">
              <span
                v-for="action in blurb.actions"
                :key="action.method || action.action"
                class="action-chip"
              >{{ formatAction(action) }}</span>
            </div>
          </div>
        </Transition>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  column: { type: String, required: true },
  phase:  { type: String, default: 'eda' },   // 'eda' | 'ml'
  tasks:  { type: Object, default: () => ({}) },
})

// Track which blurb indices are open. Reset entirely when column changes.
const openSet = ref(new Set())
watch(() => props.column, () => { openSet.value = new Set() })

function toggle(i) {
  const next = new Set(openSet.value)
  next.has(i) ? next.delete(i) : next.add(i)
  openSet.value = next
}

/**
 * Collect guidance blurbs for the selected column and phase from ALL tasks.
 * Each upgraded task stores blurbs under task.guidance[column][phase].
 */
const blurbs = computed(() => {
  const col   = props.column
  const phase = props.phase
  const out   = []
  for (const task of Object.values(props.tasks)) {
    const colGuidance = task?.guidance?.[col]
    if (!colGuidance) continue
    const phaseBlurbs = colGuidance[phase]
    if (Array.isArray(phaseBlurbs)) out.push(...phaseBlurbs)
  }
  return out
})

function levelIcon(level) {
  return { error: '🚫', warn: '⚠️', info: 'ℹ️', good: '✅' }[level] ?? 'ℹ️'
}

function formatAction(action) {
  if (typeof action === 'string') return action
  if (action.method) return action.method
  if (action.detail) return action.detail
  return action.action
}
</script>

<style scoped>
.guidance-card {
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.no-guidance {
  color: #475569;
  font-size: 13px;
  padding: 4px 0;
}

.insights {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

/* ── Individual blurb ─────────────────────────────────────────────────────── */
.insight-item {
  border-radius: 6px;
  border-left: 3px solid;
  overflow: hidden;
}
.insight-error { background: #3d0f0f; border-color: #f87171; }
.insight-warn  { background: #3d2510; border-color: #fb923c; }
.insight-info  { background: #1e3a5f; border-color: #60a5fa; }
.insight-good  { background: #14291f; border-color: #4ade80; }

/* ── Clickable title row ──────────────────────────────────────────────────── */
.insight-trigger {
  display: flex;
  align-items: center;
  gap: 7px;
  width: 100%;
  padding: 8px 10px;
  background: none;
  border: none;
  cursor: pointer;
  text-align: left;
  color: inherit;
  border-radius: 0;     /* clipped by parent */
}

/* Darken slightly on hover to signal interactivity */
.insight-error .insight-trigger:hover { background: rgba(248,113,113,0.07); }
.insight-warn  .insight-trigger:hover { background: rgba(251,146,60, 0.07); }
.insight-info  .insight-trigger:hover { background: rgba(96, 165,250,0.07); }
.insight-good  .insight-trigger:hover { background: rgba(74, 222,128,0.07); }

/* Separator when expanded */
.insight-item.is-open .insight-trigger {
  border-bottom: 1px solid rgba(255,255,255,0.06);
}

.insight-icon  { font-size: 13px; flex-shrink: 0; }
.insight-title { font-size: 12px; font-weight: 600; flex: 1; }

.insight-error .insight-title { color: #fca5a5; }
.insight-warn  .insight-title { color: #fed7aa; }
.insight-info  .insight-title { color: #bfdbfe; }
.insight-good  .insight-title { color: #bbf7d0; }

/* ── Chevron ──────────────────────────────────────────────────────────────── */
.insight-chevron {
  font-size: 16px;
  line-height: 1;
  color: #475569;
  display: inline-block;
  transform: rotate(0deg);
  transition: transform 0.18s ease;
  flex-shrink: 0;
}
.insight-chevron.rotated { transform: rotate(90deg); }

/* ── Expandable body ──────────────────────────────────────────────────────── */
.insight-body-wrap {
  padding: 8px 10px 10px;
}

.insight-body {
  font-size: 12px;
  line-height: 1.55;
  color: #94a3b8;
  margin: 0 0 8px;
}

.insight-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.action-chip {
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 10px;
  background: #0f172a;
  color: #64748b;
  border: 1px solid #334155;
  white-space: nowrap;
}

/* ── Per-blurb slide transition ───────────────────────────────────────────── */
.blurb-slide-enter-active,
.blurb-slide-leave-active {
  transition: max-height 0.2s ease, opacity 0.15s ease;
  overflow: hidden;
}
.blurb-slide-enter-from,
.blurb-slide-leave-to  { max-height: 0;    opacity: 0; }
.blurb-slide-enter-to,
.blurb-slide-leave-from { max-height: 300px; opacity: 1; }
</style>
