<template>
  <form class="filters" aria-label="Filter findings" @submit.prevent="apply">
    <label>
      Dimension
      <select v-model="draft.dimension">
        <option value="">any</option>
        <option v-for="name in dimensions" :key="name" :value="name">{{ name }}</option>
      </select>
    </label>
    <label>
      Severity at least
      <select v-model="draft.severity">
        <option value="">any</option>
        <option v-for="name in SEVERITY_ORDER" :key="name" :value="name">{{ name }}</option>
      </select>
    </label>
    <label>
      Confidence at least
      <select v-model="draft.confidence">
        <option value="">any</option>
        <option v-for="name in CONFIDENCE_ORDER" :key="name" :value="name">{{ name }}</option>
      </select>
    </label>
    <label>Check <input v-model.trim="draft.check" type="text" /></label>
    <label>Code <input v-model.trim="draft.code" type="text" /></label>
    <label>Column <input v-model.trim="draft.column" type="text" /></label>
    <label>
      Sort by
      <select v-model="draft.sort">
        <option v-for="name in SORTS" :key="name" :value="name">{{ name }}</option>
      </select>
    </label>
    <button type="submit">Apply</button>
    <button type="button" @click="clear">Clear</button>
  </form>
</template>

<script setup lang="ts">
import { reactive, watch } from 'vue'
import { CONFIDENCE_ORDER, SEVERITY_ORDER } from '../../generated/ranks'
import { SORTS, type FindingQuery } from '../../composables/findingQuery'

const props = defineProps<{ query: FindingQuery; dimensions: string[] }>()
const emit = defineEmits<{ change: [query: FindingQuery] }>()

const draft = reactive<FindingQuery>({ ...props.query })
// Back and forward navigation changes the URL; the form follows it.
watch(
  () => props.query,
  (query) => Object.assign(draft, query),
)

// Changing a filter starts again from the first page and closes the open finding.
function apply() {
  emit('change', { ...draft, offset: 0, finding: '' })
}

function clear() {
  emit('change', {
    ...props.query,
    dimension: '',
    severity: '',
    confidence: '',
    check: '',
    code: '',
    column: '',
    sort: 'severity',
    offset: 0,
    finding: '',
  })
}
</script>

<style scoped>
.filters {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: flex-end;
}
label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
}
</style>
