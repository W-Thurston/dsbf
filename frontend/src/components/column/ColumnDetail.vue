<template>
  <div class="stack" data-section="column-detail">
    <section class="card" aria-labelledby="column-title">
      <h1 id="column-title" class="run-name">{{ column }}</h1>
      <p v-if="profile" class="muted">
        <span class="badge badge-info">{{ profile.semantic_type }}</span>
        {{ profile.physical_dtype }} · {{ profile.semantic_confidence }} confidence
      </p>
    </section>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="profile === null" class="loading">Loading…</p>
    <template v-else>
      <section class="card" aria-label="Statistics" data-section="stats">
        <dl class="strip">
          <div v-for="s in strip" :key="s.label" :data-stat="s.label">
            <dt>{{ s.label }}</dt>
            <dd>{{ s.value }}</dd>
          </div>
        </dl>
      </section>

      <div class="chart-row">
        <div>
          <DsbfChart
            v-if="profile.semantic_type === 'numeric'"
            :title="view === 'box' ? 'Box plot' : 'Distribution'"
            :subtitle="subtitle"
            :theme="theme"
            :option="numericOption"
            :empty="!profile.histogram"
            empty-message="No histogram: the column has no finite values."
            :height="view === 'box' ? '160px' : '300px'"
            :export-name="`${column}-distribution`"
            data-chart="distribution"
          >
            <template #actions>
              <button
                v-if="bimodal?.bimodal && view === 'histogram'"
                class="toggle"
                :aria-pressed="showComponents"
                @click="showComponents = !showComponents"
              >
                Show fitted components
              </button>
              <button
                class="toggle"
                :aria-pressed="view === 'box'"
                @click="view = view === 'box' ? 'histogram' : 'box'"
              >
                {{ view === 'box' ? 'Histogram' : 'Box plot' }}
              </button>
            </template>
          </DsbfChart>
          <DsbfChart
            v-else-if="topValues.length"
            title="Top values"
            :theme="theme"
            :option="topValuesOption(topValues, theme)"
            :height="`${Math.max(topValues.length * 26 + 40, 120)}px`"
            :export-name="`${column}-top-values`"
            data-chart="top-values"
          />
          <section v-else class="card">
            <p class="muted">This type of column has no distribution chart.</p>
          </section>
          <p v-if="bimodal?.bimodal" class="callout" data-role="bimodal-callout" role="note">
            <strong>Bimodal.</strong> {{ bimodalText(bimodal) }}
          </p>
        </div>

        <section class="card" aria-labelledby="guidance-title" data-section="guidance">
          <h2 id="guidance-title" class="card-title">Findings and guidance</h2>
          <p v-if="findings === null" class="loading">Loading…</p>
          <p v-else-if="!findings.items.length" class="muted">No findings about this column.</p>
          <ul v-else class="findings">
            <li v-for="item in findings.items" :key="item.id">
              <details>
                <summary>
                  <SeverityBadge :severity="item.severity" />
                  <strong>{{ item.title }}</strong>
                </summary>
                <ul v-if="item.guidance?.actions.length" class="actions">
                  <li v-for="a in item.guidance.actions" :key="a.action + a.detail">
                    <span class="badge badge-ok">{{ a.action }}</span> {{ a.detail }}
                    <span v-if="a.condition" class="muted"> ({{ a.condition }})</span>
                  </li>
                </ul>
                <router-link
                  :to="{ name: 'findings', params: { runId }, query: { finding: item.id } }"
                  >Open in the findings explorer</router-link
                >
              </details>
            </li>
          </ul>
          <p v-if="findings && findings.total > findings.items.length">
            <router-link :to="{ name: 'findings', params: { runId }, query: { column } }"
              >All {{ findings.total }} findings</router-link
            >
          </p>
        </section>
      </div>

      <section v-if="rows.length" class="card" aria-label="Values">
        <details :open="numeric ? undefined : true">
          <summary class="card-title">{{ numeric ? 'Percentiles' : 'Value counts' }}</summary>
          <table class="data-table" data-section="values-table">
            <tbody>
              <tr v-for="r in rows" :key="r.label">
                <th scope="row">{{ r.label }}</th>
                <td class="num">{{ r.value }}</td>
                <td v-if="r.share" class="num">{{ r.share }}</td>
              </tr>
            </tbody>
          </table>
        </details>
      </section>

      <section v-if="lengths" class="card" aria-label="Text lengths" data-section="lengths">
        <h2 class="card-title">Text lengths</h2>
        <p>
          {{ lengths.min }} to {{ lengths.max }} characters, mean {{ lengths.mean.toFixed(1) }}.
        </p>
      </section>

      <template v-if="numeric">
        <section
          v-if="outlier"
          class="card"
          aria-labelledby="outlier-title"
          data-section="outliers"
        >
          <h2 id="outlier-title" class="card-title">Outliers</h2>
          <p
            class="callout"
            :data-agreement="
              outlier.consensus ? 'agree' : outlier.methods_flagging.length ? 'split' : 'none'
            "
            data-role="consensus"
          >
            {{ outlierConsensus(outlier) }}
          </p>
          <table class="data-table">
            <thead>
              <tr>
                <th scope="col">Method</th>
                <th scope="col">Flagged</th>
                <th scope="col">Share</th>
                <th scope="col">Rule</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in outlierRows(outlier)" :key="r.method">
                <th scope="row">{{ r.method }}</th>
                <td class="num">{{ r.count }}</td>
                <td class="num">{{ r.share }}</td>
                <td>{{ r.rule }}</td>
              </tr>
            </tbody>
          </table>
        </section>
        <section v-else class="card" data-section="outliers">
          <h2 class="card-title">Outliers</h2>
          <p class="muted">{{ reason(data.outliers, 'outliers') }}</p>
        </section>

        <section class="card" aria-labelledby="normality-title" data-section="normality">
          <h2 id="normality-title" class="card-title">Normality</h2>
          <template v-if="normal">
            <p>
              <span class="badge" :class="normal.non_normal ? 'badge-warn' : 'badge-ok'">{{
                normalitySummary(normal).verdict
              }}</span>
              <span v-for="t in normalitySummary(normal).tests" :key="t.name" class="test">
                {{ t.name }}: p {{ t.p }}
              </span>
            </p>
          </template>
          <p v-else class="muted">{{ reason(data.normality, 'normality') }}</p>
          <DsbfChart
            v-if="qq && 'qq' in qq"
            title="Q-Q plot"
            subtitle="Column quantiles against a normal fitted to it"
            :theme="theme"
            :option="qqOption(qq.qq)"
            :export-name="`${column}-qq`"
            data-chart="qq"
          />
          <p v-else-if="normal" class="muted">{{ qq && 'reason' in qq ? qq.reason : '' }}</p>
        </section>
      </template>

      <section v-if="correlations.length" class="card" aria-labelledby="corr-title">
        <h2 id="corr-title" class="card-title">Top correlations</h2>
        <ul class="correlations" data-section="correlations">
          <li v-for="c in correlations" :key="c.other">
            <router-link
              :to="{ name: 'column', params: { runId, column: c.other }, query: route.query }"
              >{{ c.other }}</router-link
            >
            <span class="muted">{{ c.statistic.replace(/_/g, ' ') }}</span>
            <strong>{{ c.value.toFixed(2) }}</strong>
          </li>
        </ul>
      </section>

      <ColumnSample :run-id="runId" :column="column" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { api, unwrap } from '../../api/client'
import type { useDistributionsData } from '../../composables/useDistributionsData'
import type { components } from '../../generated/api'
import { useThemeStore } from '../../stores/theme'
import DsbfChart from '../charts/DsbfChart.vue'
import SeverityBadge from '../SeverityBadge.vue'
import {
  columnBoxOption,
  distributionOption,
  qqOf,
  qqOption,
  topValuesOption,
} from './chartOptions'
import ColumnSample from './ColumnSample.vue'
import {
  bimodalText,
  type Profile,
  componentCurves,
  normalitySummary,
  outlierConsensus,
  outlierRows,
  percentileRows,
  statStrip,
  topCorrelations,
  valueCountRows,
} from './distributions'

const props = defineProps<{
  runId: string
  column: string
  data: ReturnType<typeof useDistributionsData>
}>()

const route = useRoute()
const theme = computed(() => useThemeStore().theme)
const profile = ref<Profile | null>(null)
const findings = ref<components['schemas']['Page_Finding_'] | null>(null)
const error = ref('')
const view = ref<'histogram' | 'box'>('histogram')
const showComponents = ref(false)

const numeric = computed(() => profile.value?.semantic_type === 'numeric')
const outlier = computed(() => props.data.outliers.value?.columns[props.column] ?? null)
const normal = computed(() => props.data.normality.value?.columns[props.column] ?? null)
const bimodal = computed(() => props.data.bimodality.value?.columns[props.column] ?? null)
const qq = computed(() =>
  normal.value
    ? qqOf(props.data.normality.value as unknown as Record<string, unknown>, props.column)
    : null,
)
const correlations = computed(() => topCorrelations(props.data.cells.value, props.column))

const strip = computed(() =>
  profile.value ? statStrip(profile.value, outlier.value?.iqr?.count ?? null) : [],
)
const topValues = computed(() =>
  profile.value?.semantic_type === 'categorical' ? profile.value.top_values : [],
)
const lengths = computed(() =>
  profile.value?.semantic_type === 'categorical' ? profile.value.lengths : null,
)
const rows = computed<{ label: string; value: string; share?: string }[]>(() => {
  const p = profile.value
  if (!p) return []
  if (p.semantic_type === 'numeric') return percentileRows(p)
  if (p.semantic_type !== 'categorical') return []
  return valueCountRows(p).map((r) => ({ label: r.label, value: r.count, share: r.share }))
})
const subtitle = computed(() => {
  const h = profile.value?.semantic_type === 'numeric' ? profile.value.histogram : null
  return h && view.value === 'histogram' ? `${h.counts.length} equal-width bins (${h.rule})` : ''
})

const numericOption = computed(() => {
  const p = profile.value
  if (!p || p.semantic_type !== 'numeric') return {}
  const q = (x: number) => p.stats?.quantiles.find((v) => v.p === x)?.value
  if (view.value === 'box') {
    const [min, q1, median, q3, max] = [p.stats?.min, q(0.25), q(0.5), q(0.75), p.stats?.max]
    if (min == null || q1 == null || median == null || q3 == null || max == null) return {}
    const iqr = outlier.value?.iqr
    return columnBoxOption(
      { min, q1, median, q3, max },
      iqr ? { lower: iqr.lower_fence, upper: iqr.upper_fence } : null,
      theme.value,
    )
  }
  const curves =
    showComponents.value && bimodal.value?.bimodal && p.histogram
      ? componentCurves(bimodal.value, p.histogram.edges, p.histogram.counts)
      : []
  return distributionOption(
    p.histogram,
    theme.value,
    { mean: p.stats?.mean ?? null, median: q(0.5) ?? null },
    curves,
  )
})

/** Why a check's card is empty: it did not run, or it skipped this column. */
function reason(
  output: { value: { skipped: Record<string, string> } | null },
  name: string,
): string {
  const why = output.value?.skipped[props.column]
  if (why) return `The ${name} check skipped this column: ${why}`
  return output.value
    ? `The ${name} check did not test this column.`
    : `The ${name} check did not run in this run.`
}

async function load() {
  profile.value = null
  findings.value = null
  error.value = ''
  view.value = 'histogram'
  showComponents.value = false
  try {
    const [p, f] = await Promise.all([
      unwrap(
        api.GET('/v2/runs/{run_id}/profile', {
          params: { path: { run_id: props.runId }, query: { column: props.column } },
        }),
      ),
      unwrap(
        api.GET('/v2/runs/{run_id}/findings', {
          params: { path: { run_id: props.runId }, query: { column: props.column, limit: 10 } },
        }),
      ),
    ])
    profile.value = p
    findings.value = f
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
.strip {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 28px;
  margin: 0;
}
.strip dt {
  color: var(--text-muted);
  font-size: 12px;
}
.strip dd {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}
.chart-row {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}
@media (max-width: 900px) {
  .chart-row {
    grid-template-columns: 1fr;
  }
}
.toggle {
  font-size: 12px;
  margin-left: 6px;
}
.toggle[aria-pressed='true'] {
  border-color: var(--link);
}
.callout[data-agreement='agree'] {
  border-left-color: var(--level-red);
}
.callout[data-agreement='none'] {
  border-left-color: var(--level-green);
}
.callout {
  margin: 12px 0 0;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--level-amber);
  border-radius: 8px;
  background: var(--surface);
}
.findings,
.actions,
.correlations {
  list-style: none;
  padding: 0;
  margin: 0;
}
.findings > li {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}
.actions {
  margin: 8px 0;
}
.actions li {
  padding: 2px 0;
}
.correlations li {
  display: grid;
  grid-template-columns: 1fr auto 56px;
  gap: 12px;
  padding: 4px 0;
}
.test {
  margin-right: 14px;
}
summary {
  cursor: pointer;
}
</style>
