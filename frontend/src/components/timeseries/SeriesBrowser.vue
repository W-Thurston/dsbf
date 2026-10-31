<template>
  <nav class="card browser" aria-label="Columns" data-section="column-browser">
    <p class="muted" role="status">{{ plural(names.length, 'column') }}</p>
    <ul>
      <li v-for="name in names" :key="name">
        <button
          type="button"
          :class="{ active: name === selected }"
          :aria-current="name === selected ? 'true' : undefined"
          :data-column="name"
          @click="$emit('select', name)"
        >
          <span class="name">{{ name }}</span>
          <span
            v-if="assessments[name]"
            :class="['badge', assessmentClass(assessments[name])]"
            data-role="assessment"
            >{{ assessmentWords(assessments[name]) }}</span
          >
        </button>
      </li>
    </ul>
  </nav>
</template>

<script setup lang="ts">
import { plural } from '../../format'
import { assessmentClass, assessmentWords } from './timeseries'

defineProps<{
  names: string[]
  selected: string
  /** Each column's stationarity assessment, where that check ran. */
  assessments: Record<string, string>
}>()
defineEmits<{ select: [column: string] }>()
</script>

<style scoped>
.browser {
  padding: 12px;
  max-height: calc(100vh - 160px);
  overflow-y: auto;
  position: sticky;
  top: 12px;
}
ul {
  list-style: none;
  margin: 0;
  padding: 0;
}
button {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  width: 100%;
  padding: 6px;
  border: 0;
  border-radius: 6px;
  background: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
button:hover,
button.active {
  background: var(--bg);
}
button.active .name {
  font-weight: 600;
}
.name {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
