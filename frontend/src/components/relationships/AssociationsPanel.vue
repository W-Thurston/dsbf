<template>
  <div class="layout" data-section="associations">
    <nav class="card browser" aria-label="Columns">
      <input
        v-model.trim="search"
        type="search"
        class="search"
        placeholder="Search columns"
        aria-label="Search columns"
      />
      <p class="muted" role="status">{{ plural(names.length, 'column') }}</p>
      <ul>
        <li v-for="c in names" :key="c.name">
          <a
            href="#"
            :class="{ active: c.name === column }"
            :aria-current="c.name === column ? 'true' : undefined"
            :data-column="c.name"
            @click.prevent="$emit('select-column', c.name)"
          >
            <span class="name">{{ c.name }}</span>
            <span class="muted type">{{ c.semantic_type }}</span>
            <span
              v-if="strong.get(c.name)"
              class="strong-count"
              :title="`${strong.get(c.name)} strong associations`"
              >{{ strong.get(c.name) }}</span
            >
          </a>
        </li>
      </ul>
      <p v-if="!names.length" class="muted">No columns match.</p>
    </nav>

    <section class="card" aria-labelledby="pairs-title">
      <h2 id="pairs-title" class="card-title">
        {{ column ? `Associations of ${column}` : 'Tested pairs' }}
      </h2>
      <div class="filters">
        <label
          >Statistic
          <select v-model="statistic" aria-label="Statistic">
            <option value="">All</option>
            <option v-for="s in matrix.statistics" :key="s" :value="s">
              {{ STATISTIC_LABEL[s] }}
            </option>
          </select>
        </label>
        <label
          >Strength
          <select v-model="strength" aria-label="Strength">
            <option value="">All</option>
            <option v-for="s in strengths" :key="s" :value="s">{{ s }}</option>
          </select>
        </label>
      </div>
      <p class="muted" role="status">{{ plural(rows.length, 'pair') }}</p>
      <div class="scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">Pair</th>
              <th scope="col">Statistic</th>
              <th scope="col">Value</th>
              <th scope="col">Strength</th>
              <th scope="col">n</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="cell in rows"
              :key="`${cell.check_id}|${cell.first}|${cell.second}|${cell.statistic}`"
            >
              <td>
                <router-link
                  :to="{ query: { ...route.query, first: cell.first, second: cell.second } }"
                  >{{ cell.first }} × {{ cell.second }}</router-link
                >
              </td>
              <td>{{ STATISTIC_LABEL[cell.statistic] }}</td>
              <td>
                {{ cell.value.toFixed(3)
                }}<template v-if="cell.interval">
                  ({{ cell.interval.low.toFixed(2) }} to
                  {{ cell.interval.high.toFixed(2) }})</template
                >
              </td>
              <td>{{ cell.strength ?? (cell.flagged ? 'flagged' : '') }}</td>
              <td>{{ formatCount(cell.n_used) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { formatCount, plural } from '../../format'
import type { components } from '../../generated/api'
import {
  associationsOf,
  sortedCells,
  STATISTIC_LABEL,
  STRENGTHS,
  strongCounts,
  touches,
} from './relationships'

const props = defineProps<{
  matrix: components['schemas']['RelationshipMatrix']
  columns: components['schemas']['ColumnSummary'][]
  column: string
  /** Columns a warning chip narrows the tables to, or null. */
  restrict: Set<string> | null
}>()
defineEmits<{ 'select-column': [name: string] }>()

const route = useRoute()
const router = useRouter()
const strengths = STRENGTHS

// Filters live in the URL, so a filtered table survives a reload and can be shared.
function param(key: string) {
  return computed({
    get: () => String(route.query[key] ?? ''),
    set: (value: string) =>
      void router.replace({ query: { ...route.query, [key]: value || undefined } }),
  })
}
const search = param('q')
const statistic = param('statistic')
const strength = param('strength')

const strong = computed(() => strongCounts(props.matrix))
const inMatrix = computed(() => new Set(props.matrix.columns))
const names = computed(() =>
  props.columns.filter(
    (c) =>
      inMatrix.value.has(c.name) &&
      c.name.toLowerCase().includes(search.value.toLowerCase()) &&
      (props.restrict === null || props.restrict.has(c.name)),
  ),
)
const rows = computed(() =>
  (props.column
    ? associationsOf(props.matrix, props.column).map((a) => a.cell)
    : sortedCells(props.matrix)
  ).filter(
    (c) =>
      (!statistic.value || c.statistic === statistic.value) &&
      (!strength.value || c.strength === strength.value) &&
      touches(c, props.restrict),
  ),
)
</script>

<style scoped>
.layout {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
@media (max-width: 900px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
.browser {
  padding: 12px;
  max-height: 520px;
  overflow-y: auto;
}
.search {
  width: 100%;
  box-sizing: border-box;
}
ul {
  list-style: none;
  margin: 0;
  padding: 0;
}
a {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 8px;
  padding: 4px 6px;
  border-radius: 6px;
  color: inherit;
  text-decoration: none;
}
a:hover,
a.active {
  background: var(--bg);
}
a.active {
  font-weight: 600;
}
.name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.type {
  font-size: 11px;
}
.strong-count {
  min-width: 18px;
  text-align: center;
  border-radius: 9px;
  background: var(--level-red, #f87171);
  color: #111;
  font-size: 11px;
}
.scroll {
  max-height: 520px;
  overflow-y: auto;
}
.filters {
  display: flex;
  gap: 16px;
  margin-bottom: 8px;
}
</style>
