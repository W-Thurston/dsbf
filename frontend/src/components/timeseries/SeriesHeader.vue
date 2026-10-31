<template>
  <section class="card" aria-label="Series" data-role="series">
    <h2 class="card-title">Series</h2>
    <dl class="strip">
      <div>
        <dt>Index column</dt>
        <dd data-role="index">
          <code>{{ series.index }}</code>
        </dd>
      </div>
      <div>
        <dt>Frequency</dt>
        <dd data-role="frequency">
          {{ series.frequency }}
          <span class="muted">({{ series.frequency_source }})</span>
        </dd>
      </div>
      <div>
        <dt>Periods</dt>
        <dd data-role="periods">
          {{ series.periods_total }}
          <span v-if="series.truncated" class="muted"
            >(the latest {{ series.max_periods }} are used)</span
          >
        </dd>
      </div>
      <div>
        <dt>Each period is the</dt>
        <dd>{{ series.aggregate }} of its rows</dd>
      </div>
      <div v-if="series.timezone">
        <dt>Time zone</dt>
        <dd>{{ series.timezone }}</dd>
      </div>
      <div>
        <dt>Value columns</dt>
        <dd data-role="value-columns">{{ columns }}</dd>
      </div>
    </dl>
    <ul v-if="notRun.length" class="muted" data-role="not-run">
      <li v-for="item in notRun" :key="item.id">{{ item.id }}: {{ item.reason }}</li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import type { SeriesOutput } from './timeseries'

defineProps<{
  series: SeriesOutput
  columns: number
  notRun: { id: string; reason: string }[]
}>()
</script>

<style scoped>
.strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px 24px;
  margin: 0;
}
dt {
  font-size: 12px;
  color: var(--text-muted);
}
dd {
  margin: 2px 0 0;
  font-weight: 600;
}
</style>
