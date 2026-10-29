<template>
  <section class="card" aria-labelledby="finding-title">
    <p class="muted"><button type="button" @click="emit('close')">Close</button></p>
    <h2 id="finding-title" class="card-title">
      <SeverityBadge :severity="finding.severity" /> {{ finding.title }}
    </h2>
    <p>{{ finding.message }}</p>
    <p class="muted">
      {{ finding.dimension }} · <code>{{ finding.code }}</code> · check
      <code>{{ finding.check_id }}</code> · confidence {{ finding.confidence }}
      <template v-if="finding.subject.columns.length">
        · {{ finding.subject.columns.join(' and ') }}
      </template>
    </p>

    <h3>Evidence</h3>
    <p>
      Method <code>{{ evidence.method }}</code> on {{ formatCount(evidence.n_used) }} of
      {{ formatCount(evidence.n_total) }} rows<template v-if="evidence.interval"
        >, {{ evidence.interval.level * 100 }}% interval {{ evidence.interval.low }} to
        {{ evidence.interval.high }}</template
      >.
    </p>
    <SamplingNotice v-if="evidence.sampled" :sampling="evidence.sampled" />
    <table v-if="metrics.length" class="data-table">
      <tbody>
        <tr v-for="[name, value] in metrics" :key="name">
          <th scope="row">{{ name }}</th>
          <td>{{ value }}</td>
        </tr>
      </tbody>
    </table>
    <p v-if="evidence.non_finite_metrics.length" class="muted">
      Not finite: {{ evidence.non_finite_metrics.join(', ') }}.
    </p>
    <template v-if="evidence.assumptions.length">
      <h3>Assumptions</h3>
      <ul>
        <li v-for="item in evidence.assumptions" :key="item.name">
          {{ item.name }}: {{ describe(item.satisfied)
          }}<span v-if="item.detail"> ({{ item.detail }})</span>
        </li>
      </ul>
    </template>

    <template v-if="finding.guidance?.actions.length">
      <h3>What to do</h3>
      <ul>
        <li v-for="(item, index) in finding.guidance.actions" :key="index">
          <strong>{{ words(item.action) }}</strong>
          <span v-if="item.columns.length"> {{ item.columns.join(', ') }}</span>
          <span v-if="item.condition" class="muted"> if {{ item.condition }}</span>
          <span v-if="item.detail"> — {{ item.detail }}</span>
        </li>
      </ul>
    </template>

    <template v-if="evidence.details_ref">
      <h3>Full detail</h3>
      <ArtifactViewer :run-id="runId" :artifact-ref="evidence.details_ref" />
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { formatCount, words } from '../../format'
import type { components } from '../../generated/api'
import SamplingNotice from '../run/SamplingNotice.vue'
import SeverityBadge from '../SeverityBadge.vue'
import ArtifactViewer from './ArtifactViewer.vue'

const props = defineProps<{ runId: string; finding: components['schemas']['Finding'] }>()
const emit = defineEmits<{ close: [] }>()

const evidence = computed(() => props.finding.evidence)
const metrics = computed(() =>
  Object.entries(evidence.value.metrics).map(
    ([name, value]) =>
      [
        name,
        typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value),
      ] as const,
  ),
)

function describe(satisfied: boolean | null): string {
  return satisfied === null ? 'not checked' : satisfied ? 'met' : 'not met'
}
</script>
