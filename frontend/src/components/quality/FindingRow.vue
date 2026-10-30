<template>
  <tr class="row" :data-finding="finding.id">
    <td>
      <button
        type="button"
        class="toggle"
        :aria-expanded="open"
        :aria-label="`${open ? 'Hide' : 'Show'} the evidence for ${finding.title}`"
        @click="toggle"
      >
        {{ open ? '▾' : '▸' }}
      </button>
    </td>
    <td><SeverityBadge :severity="finding.severity as SeverityName" /></td>
    <td>
      <router-link
        v-for="column in finding.columns"
        :key="column"
        class="column"
        :to="{ name: 'column', params: { runId, column } }"
        >{{ column }}</router-link
      >
      <span v-if="!finding.columns.length" class="muted">dataset</span>
    </td>
    <td>{{ finding.title }}</td>
  </tr>
  <tr v-if="open" class="detail" data-role="finding-detail">
    <td colspan="4">
      <p v-if="error" class="error">{{ error }}</p>
      <p v-else-if="!full" class="loading">Loading…</p>
      <template v-else>
        <p>{{ full.message }}</p>
        <dl v-if="metrics.length" class="metrics">
          <template v-for="[name, value] in metrics" :key="name">
            <dt>{{ name }}</dt>
            <dd>{{ value }}</dd>
          </template>
        </dl>
        <ul v-if="full.guidance?.actions.length" class="guidance">
          <li v-for="(item, index) in full.guidance.actions" :key="index">
            <strong>{{ words(item.action) }}</strong>
            <span v-if="item.condition" class="muted"> if {{ item.condition }}</span>
            <span v-if="item.detail"> — {{ item.detail }}</span>
          </li>
        </ul>
        <router-link
          :to="{
            name: 'findings',
            params: { runId },
            query: { finding: finding.id, check: finding.check_id },
          }"
          >Open in the findings explorer</router-link
        >
      </template>
    </td>
  </tr>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { api, unwrap } from '../../api/client'
import { words } from '../../format'
import type { components } from '../../generated/api'
import type { SeverityName } from '../../generated/ranks'
import SeverityBadge from '../SeverityBadge.vue'
import type { FindingRef } from './quality'

const props = defineProps<{ runId: string; finding: FindingRef }>()

const open = ref(false)
const full = ref<components['schemas']['Finding'] | null>(null)
const error = ref('')

// The evidence is fetched when the row first opens, so Quality does not carry every
// finding's detail up front.
async function toggle() {
  open.value = !open.value
  if (!open.value || full.value) return
  try {
    full.value = await unwrap(
      api.GET('/v2/runs/{run_id}/findings/{finding_id}', {
        params: { path: { run_id: props.runId, finding_id: props.finding.id } },
      }),
    )
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

const metrics = computed(() =>
  Object.entries(full.value?.evidence.metrics ?? {}).map(
    ([name, value]) =>
      [
        words(name),
        typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value),
      ] as const,
  ),
)
</script>

<style scoped>
.toggle {
  border: 0;
  background: none;
  cursor: pointer;
  color: inherit;
  padding: 0 4px;
}
.column + .column::before {
  content: ', ';
  color: var(--text-muted);
}
.detail td {
  background: var(--bg);
  padding: 10px 16px 14px;
}
.metrics {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 2px 16px;
  font-size: 13px;
}
.metrics dt {
  color: var(--text-muted);
}
.metrics dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}
.guidance {
  padding-left: 18px;
  font-size: 13px;
}
</style>
