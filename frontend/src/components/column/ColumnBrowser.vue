<template>
  <nav class="card browser" aria-label="Columns" data-section="column-browser">
    <input
      v-model.trim="search"
      type="search"
      class="search"
      placeholder="Search columns"
      aria-label="Search columns"
    />
    <p class="muted" role="status">{{ plural(count, 'column') }}</p>
    <section v-for="group in groups" :key="group.type" :data-group="group.type">
      <h2 class="card-title">{{ group.type }} · {{ group.columns.length }}</h2>
      <ul>
        <li v-for="c in group.columns" :key="c.name">
          <router-link
            :to="{ name: 'column', params: { runId, column: c.name }, query: route.query }"
            :class="{ active: c.name === selected }"
            :aria-current="c.name === selected ? 'page' : undefined"
            :data-column="c.name"
          >
            <span class="name">{{ c.name }}</span>
            <span
              v-if="c.warned"
              class="dot"
              role="img"
              aria-label="has a warning or error finding"
              data-role="warning-dot"
            />
            <span
              class="null-bar"
              data-role="null-bar"
              :title="`${(c.nullShare * 100).toFixed(1)}% null`"
            >
              <span :style="{ width: `${Math.min(c.nullShare * 100, 100)}%` }" />
            </span>
          </router-link>
        </li>
      </ul>
    </section>
    <p v-if="!count" class="muted">No columns match.</p>
  </nav>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { plural } from '../../format'
import type { components } from '../../generated/api'
import { browserGroups } from './distributions'

const props = defineProps<{
  runId: string
  columns: components['schemas']['ColumnSummary'][]
  warned: Set<string>
  selected: string
}>()

const route = useRoute()
const router = useRouter()

// The search lives in the URL, so a filtered browser survives a reload and can be shared.
const search = computed({
  get: () => String(route.query.q ?? ''),
  set: (value: string) => void router.replace({ query: { ...route.query, q: value || undefined } }),
})
const groups = computed(() => browserGroups(props.columns, props.warned, search.value))
const count = computed(() => groups.value.reduce((n, g) => n + g.columns.length, 0))
</script>

<style scoped>
.browser {
  padding: 12px;
  max-height: calc(100vh - 160px);
  overflow-y: auto;
  position: sticky;
  top: 12px;
}
.search {
  width: 100%;
  box-sizing: border-box;
}
ul {
  list-style: none;
  margin: 0 0 12px;
  padding: 0;
}
a {
  display: grid;
  grid-template-columns: 1fr auto 44px;
  align-items: center;
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
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--level-amber, #fbbf24);
}
.null-bar {
  grid-column: 3;
  height: 6px;
  border-radius: 3px;
  background: var(--border);
  overflow: hidden;
}
.null-bar span {
  display: block;
  height: 100%;
  background: var(--level-red, #f87171);
}
</style>
