<template>
  <section class="card" aria-labelledby="health-title">
    <h2 id="health-title" class="card-title">Run health</h2>
    <ul class="status-counts">
      <li v-for="status in statuses" :key="status">
        <span :class="['badge', statusClass(status)]">{{ words(status) }}</span>
        <span class="count">{{ health.counts[status] }}</span>
      </li>
    </ul>

    <p v-if="!health.issues.length && !notes.length" class="muted">
      Every check ran to completion.
    </p>

    <template v-for="status in issueStatuses" :key="status">
      <h3 class="group">{{ words(status) }}</h3>
      <ul class="issues" :data-status="status">
        <li v-for="issue in issuesWith(status)" :key="issue.check_id">
          <code>{{ issue.check_id }}</code>
          <p>{{ issue.message }}</p>
          <p v-if="issue.hint" class="muted">Hint: {{ issue.hint }}</p>
          <p v-if="issue.blocked_by.length" class="muted">
            Blocked by <code>{{ issue.blocked_by.join(', ') }}</code>
          </p>
        </li>
      </ul>
    </template>

    <template v-if="notes.length">
      <h3 class="group">Run notes</h3>
      <ul class="issues" data-status="notes">
        <li v-for="note in notes" :key="note.code">
          <code>{{ note.code }}</code>
          <SeverityBadge :severity="note.severity" />
          <p>{{ note.message }}</p>
          <p v-if="note.hint" class="muted">Hint: {{ note.hint }}</p>
        </li>
      </ul>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { components } from '../../generated/api'
import { words } from '../../format'
import SeverityBadge from '../SeverityBadge.vue'

type Health = components['schemas']['RunHealth']
type Status = components['schemas']['CheckStatus']

const props = defineProps<{ health: Health; notes: components['schemas']['Note'][] }>()

const statuses: Status[] = ['ok', 'not_applicable', 'unavailable', 'skipped', 'error', 'timeout']
// The order a reader needs: what failed first, what was merely not applicable last.
const ISSUE_ORDER: Status[] = ['error', 'timeout', 'skipped', 'unavailable', 'not_applicable']

const issueStatuses = computed(() =>
  ISSUE_ORDER.filter((status) => props.health.issues.some((issue) => issue.status === status)),
)

function issuesWith(status: Status) {
  return props.health.issues.filter((issue) => issue.status === status)
}

function statusClass(status: Status): string {
  if (status === 'ok') return 'badge-ok'
  if (status === 'error' || status === 'timeout') return 'badge-error'
  if (status === 'not_applicable') return 'badge-info'
  return 'badge-warn'
}
</script>

<style scoped>
.status-counts {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 20px;
  list-style: none;
  padding: 0;
  margin: 0 0 12px;
}
.count {
  margin-left: 6px;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}
.group {
  font-size: 13px;
  text-transform: capitalize;
  margin: 16px 0 6px;
}
.issues {
  list-style: none;
  padding: 0;
  margin: 0;
}
.issues li {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}
.issues p {
  margin: 2px 0 0;
}
</style>
