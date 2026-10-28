<template>
  <section class="card" aria-labelledby="counts-title">
    <h2 id="counts-title" class="card-title">Findings by dimension and severity</h2>
    <p v-if="!counts.length" class="muted">No findings.</p>
    <table v-else class="data-table">
      <thead>
        <tr>
          <th scope="col">Dimension</th>
          <th v-for="severity in severities" :key="severity" scope="col" class="num">
            {{ severity }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="dimension in dimensions" :key="dimension">
          <th scope="row">{{ dimension }}</th>
          <td v-for="severity in severities" :key="severity" class="num">
            {{ countOf(dimension, severity) || '–' }}
          </td>
        </tr>
      </tbody>
    </table>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { components } from '../../generated/api'
import { SEVERITY_ORDER, type SeverityName } from '../../generated/ranks'

type Counts = components['schemas']['FindingCount'][]

const props = defineProps<{ counts: Counts }>()

// Most severe first, so the eye lands on the left.
const severities = [...SEVERITY_ORDER].reverse()
const dimensions = computed(() => [...new Set(props.counts.map((c) => c.dimension))].sort())

function countOf(dimension: string, severity: SeverityName): number {
  return props.counts
    .filter((c) => c.dimension === dimension && c.severity === severity)
    .reduce((sum, c) => sum + c.count, 0)
}
</script>
