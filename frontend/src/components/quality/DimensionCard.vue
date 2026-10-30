<template>
  <section class="card dimension" :aria-labelledby="`${id}-title`" :data-dimension="id">
    <header class="head">
      <h3 :id="`${id}-title`" class="dimension-title">{{ label }}</h3>
      <span :class="['badge', levelClass(level)]" data-role="level">{{ levelWords(level) }}</span>
    </header>
    <p v-if="!assessed" class="muted" data-role="not-assessed">
      {{ unassessedReason(inputs) }}
    </p>
    <p v-else-if="!findings.length" class="muted">No findings.</p>
    <p class="muted">Checks read: {{ percent(coverage) }}</p>
    <ul class="inputs">
      <li v-for="input in inputs" :key="input.check_id">
        <code>{{ input.check_id }}</code> {{ inputWords(input.state)
        }}<template v-if="input.state === 'assessed'"
          >,
          <router-link
            v-if="input.findings"
            :to="{ name: 'findings', params: { runId }, query: { check: input.check_id } }"
            >{{ input.findings }} finding(s)</router-link
          ><template v-else>0 findings</template></template
        >
      </li>
    </ul>
    <ul v-if="findings.length" class="findings">
      <li v-for="finding in findings" :key="finding.id">
        <SeverityBadge :severity="finding.severity as SeverityName" />
        <router-link
          :to="{
            name: 'findings',
            params: { runId },
            query: { finding: finding.id, check: finding.check_id },
          }"
          >{{ finding.title }}</router-link
        >
        <span v-if="finding.columns.length" class="muted"> ({{ finding.columns.join(', ') }})</span>
        <ul v-if="finding.actions?.length" class="actions">
          <li v-for="(action, index) in finding.actions" :key="index">
            {{ actionWords(action) }}
          </li>
        </ul>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import SeverityBadge from '../SeverityBadge.vue'
import type { SeverityName } from '../../generated/ranks'
import {
  actionWords,
  inputWords,
  levelClass,
  levelWords,
  percent,
  unassessedReason,
  type FindingRef,
  type InputReport,
  type Level,
} from './quality'

defineProps<{
  id: string
  runId: string
  label: string
  assessed: boolean
  level: Level | null
  coverage: number
  inputs: InputReport[]
  findings: FindingRef[]
}>()
</script>

<style scoped>
.head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.dimension-title {
  font-size: 15px;
  text-transform: capitalize;
}
.inputs,
.findings,
.actions {
  padding-left: 18px;
  font-size: 13px;
}
.findings > li {
  margin: 6px 0;
}
</style>
