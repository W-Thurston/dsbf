<template>
  <section class="card" aria-labelledby="columns-title" data-section="columns">
    <h2 id="columns-title" class="card-title">Columns</h2>
    <div class="controls">
      <input
        v-model="search"
        type="search"
        placeholder="Filter columns"
        aria-label="Filter columns by name"
      />
      <select v-model="type" aria-label="Filter columns by type">
        <option value="">All types</option>
        <option v-for="t in types" :key="t" :value="t">{{ t }}</option>
      </select>
      <span class="muted count">{{ shown.length }} of {{ columns.length }}</span>
    </div>
    <div class="scroll">
      <table class="data-table">
        <thead>
          <tr>
            <th
              v-for="head in heads"
              :key="head.label"
              scope="col"
              :class="{ num: head.num }"
              :aria-sort="
                head.by && head.by === sortBy
                  ? descending
                    ? 'descending'
                    : 'ascending'
                  : undefined
              "
            >
              <button v-if="head.by" type="button" class="sort" @click="sortOn(head.by)">
                {{ head.label }}
              </button>
              <template v-else>{{ head.label }}</template>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="column in shown" :key="column.name" :data-column="column.name">
            <th scope="row">
              <router-link :to="{ name: 'column', params: { runId, column: column.name } }">{{
                column.name
              }}</router-link>
            </th>
            <td>
              <span class="badge badge-info">{{ column.semantic_type }}</span>
              <span class="muted dtype">{{ column.physical_dtype }}</span>
            </td>
            <td class="num" :class="`tone-${missingTone(nullShare(column))}`" data-cell="missing">
              {{ percent(nullShare(column)) }}
            </td>
            <td class="num">{{ formatCount(column.counts.distinct) }}</td>
            <td>
              <span v-if="column.sketch.mode !== null" class="mode">{{ column.sketch.mode }}</span>
              <span v-else class="muted">–</span>
            </td>
            <td class="num" :class="`tone-${modeTone(column.sketch.mode_share)}`" data-cell="mode">
              {{ column.sketch.mode_share === null ? '–' : percent(column.sketch.mode_share) }}
            </td>
            <td class="num" :class="`tone-${skewTone(column.sketch.skewness)}`" data-cell="skew">
              {{ formatFigure(column.sketch.skewness) }}
            </td>
            <td class="num">
              {{
                column.sketch.min === null
                  ? '–'
                  : `${formatFigure(column.sketch.min)} to ${formatFigure(column.sketch.max)}`
              }}
            </td>
            <td>
              <RowSketch
                :bars="sketchBars(column)"
                :label="`${column.name}: ${column.sketch.bins.length ? 'histogram' : 'top values'}`"
              />
            </td>
          </tr>
          <tr v-if="!shown.length">
            <td :colspan="heads.length" class="muted">No column matches.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { formatCount } from '../../format'
import type { components } from '../../generated/api'
import RowSketch from './RowSketch.vue'
import {
  formatFigure,
  missingTone,
  modeTone,
  nullShare,
  percent,
  skewTone,
  sketchBars,
  sortColumns,
  type ColumnSort,
} from './overview'

const props = defineProps<{ runId: string; columns: components['schemas']['ColumnSummary'][] }>()

const search = ref('')
const type = ref('')
const sortBy = ref<ColumnSort>('position')
const descending = ref(false)

const heads: { label: string; by?: ColumnSort; num?: boolean }[] = [
  { label: 'Column', by: 'name' },
  { label: 'Type', by: 'type' },
  { label: 'Missing', by: 'missing', num: true },
  { label: 'Unique', by: 'unique', num: true },
  { label: 'Mode' },
  { label: 'Mode share', by: 'mode', num: true },
  { label: 'Skew', by: 'skew', num: true },
  { label: 'Range', num: true },
  { label: 'Shape' },
]

const types = computed(() => [...new Set(props.columns.map((c) => c.semantic_type))].sort())
const shown = computed(() => {
  const needle = search.value.trim().toLowerCase()
  const kept = props.columns.filter(
    (c) =>
      (!type.value || c.semantic_type === type.value) &&
      (!needle || c.name.toLowerCase().includes(needle)),
  )
  return sortColumns(kept, sortBy.value, descending.value)
})

function sortOn(by: ColumnSort) {
  descending.value = sortBy.value === by ? !descending.value : by !== 'name' && by !== 'type'
  sortBy.value = by
}
</script>

<style scoped>
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-bottom: 10px;
}
.count {
  margin-left: auto;
}
.scroll {
  overflow-x: auto;
}
.sort {
  all: unset;
  cursor: pointer;
  font-weight: 600;
}
.sort:focus-visible {
  outline: 2px solid var(--link);
}
.dtype {
  margin-left: 4px;
  font-size: 12px;
}
.mode {
  display: inline-block;
  max-width: 14ch;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  vertical-align: bottom;
}
.tone-amber {
  color: var(--level-amber);
  font-weight: 600;
}
.tone-red {
  color: var(--level-red);
  font-weight: 600;
}
td,
th {
  height: 34px;
}
</style>
