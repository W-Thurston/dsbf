<template>
  <div class="stack">
    <p v-if="!diff.same_dataset" class="notice" data-role="different-datasets">
      These runs are not of the same dataset, so the differences describe two different tables.
    </p>
    <p class="muted" data-role="diff-summary">
      {{ diff.new.length }} new, {{ diff.resolved.length }} resolved,
      {{ diff.changed.length }} changed findings; {{ diff.status_changes.length }} check status
      {{ diff.status_changes.length === 1 ? 'change' : 'changes' }} (tolerance
      {{ diff.tolerance }}).
    </p>

    <DiffCounts :diff="diff" />

    <section v-for="group in groups" :key="group.name" class="card" :data-section="group.name">
      <h2 class="card-title">{{ group.title }} ({{ group.findings.length }})</h2>
      <p v-if="!group.findings.length" class="muted">None.</p>
      <ul v-else class="findings">
        <li v-for="finding in cap(group.name, group.findings)" :key="finding.id">
          <SeverityBadge :severity="finding.severity" />
          <strong>{{ finding.title }}</strong>
          <span class="muted"> · {{ finding.check_id }} · {{ finding.code }}</span>
        </li>
      </ul>
      <button
        v-if="group.findings.length > CAP"
        type="button"
        class="more"
        :data-more="group.name"
        @click="toggle(group.name)"
      >
        {{ open.has(group.name) ? 'Show fewer' : `Show all ${group.findings.length}` }}
      </button>
    </section>

    <section class="card" data-section="changed">
      <h2 class="card-title">Changed ({{ diff.changed.length }})</h2>
      <p v-if="!diff.changed.length" class="muted">None.</p>
      <table v-else class="data-table">
        <thead>
          <tr>
            <th scope="col">Finding</th>
            <th scope="col">Severity</th>
            <th scope="col">Metrics</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="change in cap('changed', diff.changed)" :key="change.id">
            <td>
              {{ change.title }} <span class="muted">· {{ change.code }}</span>
            </td>
            <td>
              <template v-if="change.base_severity !== change.head_severity">
                <SeverityBadge :severity="change.base_severity" /> →
                <SeverityBadge :severity="change.head_severity" />
              </template>
              <SeverityBadge v-else :severity="change.head_severity" />
            </td>
            <td>
              <span v-for="metric in change.metrics" :key="metric.name" class="metric">
                {{ metric.name }} {{ show(metric.base) }} → {{ show(metric.head) }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
      <button
        v-if="diff.changed.length > CAP"
        type="button"
        class="more"
        data-more="changed"
        @click="toggle('changed')"
      >
        {{ open.has('changed') ? 'Show fewer' : `Show all ${diff.changed.length}` }}
      </button>
    </section>

    <section class="card" data-section="status-changes">
      <h2 class="card-title">Check status changes ({{ diff.status_changes.length }})</h2>
      <p v-if="!diff.status_changes.length" class="muted">None.</p>
      <table v-else class="data-table">
        <thead>
          <tr>
            <th scope="col">Check</th>
            <th scope="col">Base</th>
            <th scope="col">Head</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="change in diff.status_changes" :key="change.check_id">
            <td>{{ change.check_id }}</td>
            <td>{{ change.base ? words(change.base) : 'not run' }}</td>
            <td>{{ change.head ? words(change.head) : 'not run' }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive } from 'vue'
import { words } from '../../format'
import type { components } from '../../generated/api'
import SeverityBadge from '../SeverityBadge.vue'
import DiffCounts from './DiffCounts.vue'

const props = defineProps<{ diff: components['schemas']['RunDiff'] }>()

const groups = computed(() => [
  { name: 'new', title: 'New in head', findings: props.diff.new },
  { name: 'resolved', title: 'Resolved since base', findings: props.diff.resolved },
])

// A full scan against its sample can change hundreds of findings; show the first rows
// of each section and let the person ask for the rest.
const CAP = 25
const open = reactive(new Set<string>())
function toggle(name: string) {
  if (!open.delete(name)) open.add(name)
}
function cap<T>(name: string, rows: T[]): T[] {
  return open.has(name) ? rows : rows.slice(0, CAP)
}

function show(value: boolean | number | string | null): string {
  if (value === null) return '–'
  return typeof value === 'number' ? String(Number(value.toPrecision(4))) : String(value)
}
</script>

<style scoped>
.more {
  margin-top: 12px;
}
.metric {
  display: block;
}
.notice {
  margin: 0;
  padding: 8px 12px;
  border: 1px solid var(--border, currentColor);
  border-radius: 6px;
}
</style>
