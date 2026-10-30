<template>
  <section class="card" aria-labelledby="counts-title" data-section="finding-counts">
    <h2 id="counts-title" class="card-title">Findings by dimension and severity</h2>
    <p v-if="!counts.length" class="muted">No findings.</p>
    <template v-else>
      <ul class="rows">
        <li v-for="dimension in dimensions" :key="dimension" :data-finding-dimension="dimension">
          <span class="name">{{ dimension }}</span>
          <StrengthBar :segments="segments(dimension)" />
          <span class="total">{{ total(dimension) }}</span>
        </li>
      </ul>
      <ul class="legend" aria-label="Severity colours">
        <li v-for="severity in severities" :key="severity">
          <span class="swatch" :style="{ background: `var(--level-${SEVERITY_MARK[severity]})` }" />
          {{ severity }}
        </li>
      </ul>
      <table class="data-table sr-only" aria-label="Finding counts">
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
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { components } from '../../generated/api'
import { SEVERITY_ORDER, type SeverityName } from '../../generated/ranks'
import StrengthBar from '../charts/StrengthBar.vue'
import { SEVERITY_MARK } from '../overview/overview'

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

function segments(dimension: string) {
  return severities.map((severity) => ({
    label: severity,
    value: countOf(dimension, severity),
    mark: SEVERITY_MARK[severity] ?? 'none',
  }))
}

function total(dimension: string): number {
  return severities.reduce((sum, severity) => sum + countOf(dimension, severity), 0)
}
</script>

<style scoped>
.rows,
.legend {
  list-style: none;
  padding: 0;
  margin: 0;
}
.rows li {
  display: grid;
  grid-template-columns: 110px 1fr 40px;
  gap: 10px;
  align-items: center;
  padding: 5px 0;
}
.name {
  text-transform: capitalize;
}
.total {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 14px;
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-muted);
}
.swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  margin-right: 4px;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
}
</style>
