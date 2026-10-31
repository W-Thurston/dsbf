<template>
  <details
    :id="`${idPrefix}-section-${name}`"
    class="card section"
    :open="opensFirst"
    :data-section="name"
  >
    <summary>
      <TrafficDot :mark="dimension.level ?? 'none'" :size="12" />
      <h2 class="card-title title">{{ label }}</h2>
      <router-link
        v-if="findings.length"
        class="muted"
        :to="{ name: 'findings', params: { runId }, query: { dimension: name } }"
        @click.stop
        >{{ findings.length }} {{ findings.length === 1 ? 'finding' : 'findings' }}</router-link
      >
      <span v-else class="muted">0 findings</span>
    </summary>
    <p v-if="!dimension.assessed" class="muted" data-role="not-assessed">
      {{ unassessedReason(dimension.inputs) }}
    </p>
    <p v-else-if="!findings.length" class="muted">No findings.</p>
    <div v-else class="scroll">
      <table class="data-table">
        <thead>
          <tr>
            <th scope="col"><span class="visually-hidden">Evidence</span></th>
            <th v-for="head in heads" :key="head.by" scope="col" :aria-sort="ariaSort(head.by)">
              <button type="button" class="sort" @click="sortOn(head.by)">{{ head.label }}</button>
            </th>
          </tr>
        </thead>
        <tbody>
          <FindingRow v-for="f in findings" :key="f.id" :run-id="runId" :finding="f" />
        </tbody>
      </table>
    </div>
    <p class="muted inputs">
      Checks read: {{ percent(dimension.coverage) }}.
      <template v-for="input in dimension.inputs" :key="input.check_id">
        <code>{{ input.check_id }}</code> {{ inputWords(input.state) }}.
      </template>
    </p>
  </details>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import TrafficDot from '../shell/TrafficDot.vue'
import FindingRow from './FindingRow.vue'
import {
  inputWords,
  percent,
  sortFindings,
  unassessedReason,
  type FindingSort,
  type QualityDimension,
  type ReadinessDimension,
} from './quality'

const props = withDefaults(
  defineProps<{
    runId: string
    name: string
    label: string
    dimension: Pick<
      QualityDimension | ReadinessDimension,
      'assessed' | 'level' | 'coverage' | 'inputs' | 'findings'
    >
    idPrefix?: string
  }>(),
  { idPrefix: 'quality' },
)

// A section with findings or a reason to read opens; a clean one stays folded.
const opensFirst = props.dimension.findings.length > 0 || !props.dimension.assessed
const sortBy = ref<FindingSort>('severity')
const descending = ref(true)

const heads: { by: FindingSort; label: string }[] = [
  { by: 'severity', label: 'Severity' },
  { by: 'column', label: 'Column' },
  { by: 'finding', label: 'Finding' },
]

const findings = computed(() =>
  sortFindings(props.dimension.findings, sortBy.value, descending.value),
)

function sortOn(by: FindingSort) {
  descending.value = sortBy.value === by ? !descending.value : by === 'severity'
  sortBy.value = by
}

function ariaSort(by: FindingSort) {
  return sortBy.value === by ? (descending.value ? 'descending' : 'ascending') : undefined
}
</script>

<style scoped>
summary {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
}
.title {
  margin: 0;
  text-transform: none;
  font-size: 15px;
  color: inherit;
}
.sort {
  border: 0;
  background: none;
  font: inherit;
  font-weight: 600;
  color: var(--text-muted);
  cursor: pointer;
  padding: 0;
}
.scroll {
  overflow-x: auto;
  margin-top: 12px;
}
.inputs {
  font-size: 12px;
  margin-bottom: 0;
}
</style>
