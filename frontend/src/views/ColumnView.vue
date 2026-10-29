<template>
  <div class="stack">
    <section class="card" aria-labelledby="column-title">
      <p class="muted crumbs">
        <router-link to="/">Datasets</router-link> /
        <router-link :to="{ name: 'run', params: { runId } }">Run</router-link> /
        <router-link :to="{ name: 'columns', params: { runId } }">Columns</router-link> /
      </p>
      <h1 id="column-title" class="run-name">{{ column }}</h1>
      <p v-if="profile" class="muted">
        {{ profile.semantic_type }} ({{ profile.semantic_confidence }} confidence)
      </p>
    </section>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="profile === null" class="loading">Loading…</p>
    <template v-else>
      <section class="card" aria-labelledby="stats-title">
        <h2 id="stats-title" class="card-title">Profile</h2>
        <dl class="stats">
          <template v-for="[label, value] in statRows(profile)" :key="label">
            <dt>{{ label }}</dt>
            <dd>{{ value }}</dd>
          </template>
        </dl>
      </section>

      <DsbfChart
        v-if="profile.semantic_type === 'numeric'"
        title="Histogram"
        :subtitle="histogramSubtitle"
        :theme="theme"
        :option="histogramOption(profile.histogram)"
        :empty="!profile.histogram"
        empty-message="No histogram: the column has no finite values."
        :export-name="`${column}-histogram`"
      />

      <DsbfChart
        v-if="topValues.length"
        title="Top values"
        :theme="theme"
        :option="topValuesOption(topValues)"
        :height="`${Math.max(topValues.length * 26 + 40, 120)}px`"
        :export-name="`${column}-top-values`"
      />

      <template v-if="profile.semantic_type === 'numeric'">
        <DsbfChart
          v-if="qq && 'qq' in qq"
          title="Q-Q plot"
          subtitle="Column quantiles against a normal fitted to it"
          :theme="theme"
          :option="qqOption(qq.qq)"
          :export-name="`${column}-qq`"
        />
        <section v-else class="card" aria-labelledby="qq-title">
          <h2 id="qq-title" class="card-title">Q-Q plot</h2>
          <p class="muted">{{ qq ? qq.reason : 'Loading…' }}</p>
        </section>
      </template>

      <section class="card" aria-labelledby="column-findings-title">
        <h2 id="column-findings-title" class="card-title">Findings</h2>
        <p v-if="findings === null" class="loading">Loading…</p>
        <p v-else-if="!findings.items.length" class="muted">No findings about this column.</p>
        <ol v-else class="findings">
          <li v-for="item in findings.items" :key="item.id">
            <SeverityBadge :severity="item.severity" />
            <router-link :to="{ name: 'findings', params: { runId }, query: { finding: item.id } }">
              <strong>{{ item.title }}</strong>
            </router-link>
            <span class="muted"> · {{ item.dimension }} · {{ item.code }}</span>
          </li>
        </ol>
        <p v-if="findings && findings.total > findings.items.length">
          <router-link :to="{ name: 'findings', params: { runId }, query: { column } }"
            >All {{ findings.total }} findings</router-link
          >
        </p>
      </section>

      <ColumnSample :run-id="runId" :column="column" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../api/client'
import DsbfChart from '../components/charts/DsbfChart.vue'
import { histogramOption, qqOf, qqOption, topValuesOption } from '../components/column/chartOptions'
import ColumnSample from '../components/column/ColumnSample.vue'
import { statRows } from '../components/column/columnStats'
import SeverityBadge from '../components/SeverityBadge.vue'
import type { components } from '../generated/api'
import { useThemeStore } from '../stores/theme'

const props = defineProps<{ runId: string; column: string }>()

type Profile = Awaited<ReturnType<typeof fetchProfile>>
const theme = computed(() => useThemeStore().theme)
const profile = ref<Profile | null>(null)
const findings = ref<components['schemas']['Page_Finding_'] | null>(null)
const qq = ref<ReturnType<typeof qqOf> | null>(null)
const error = ref('')

const topValues = computed(() => {
  const p = profile.value
  return p && 'top_values' in p ? p.top_values : []
})
const histogramSubtitle = computed(() => {
  const p = profile.value
  return p && 'histogram' in p && p.histogram
    ? `${p.histogram.counts.length} equal-width bins (${p.histogram.rule})`
    : ''
})

function fetchProfile() {
  return unwrap(
    api.GET('/v2/runs/{run_id}/profile', {
      params: { path: { run_id: props.runId }, query: { column: props.column } },
    }),
  )
}

async function load() {
  profile.value = null
  findings.value = null
  qq.value = null
  error.value = ''
  try {
    profile.value = await fetchProfile()
    findings.value = await unwrap(
      api.GET('/v2/runs/{run_id}/findings', {
        params: { path: { run_id: props.runId }, query: { column: props.column, limit: 10 } },
      }),
    )
    if (profile.value.semantic_type === 'numeric') {
      try {
        const result = await unwrap(
          api.GET('/v2/runs/{run_id}/checks/{check_id}', {
            params: { path: { run_id: props.runId, check_id: 'normality' } },
          }),
        )
        qq.value = qqOf(result.output, props.column)
      } catch {
        qq.value = { reason: 'The normality check did not run in this run.' }
      }
    }
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

watch(() => [props.runId, props.column], load, { immediate: true })
</script>

<style scoped>
.run-name {
  font-size: 22px;
}
.crumbs {
  margin: 0 0 4px;
}
.stats {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 4px 16px;
  margin: 0;
}
.stats dt {
  color: var(--muted, inherit);
}
.stats dd {
  margin: 0;
}
.findings {
  list-style: none;
  padding: 0;
  margin: 0;
}
.findings li {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}
</style>
