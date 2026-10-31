<template>
  <section class="card" aria-labelledby="acf-title" data-section="autocorrelation">
    <h2 id="acf-title" class="card-title">Autocorrelation</h2>
    <p v-if="!info" class="muted">{{ column }} was not analysed: {{ skipped }}</p>
    <template v-else>
      <p class="muted" data-role="ljung-box">
        Ljung-Box p = {{ info.ljung_box_p.toPrecision(2) }}:
        {{
          info.flagged_acf.length
            ? `${info.flagged_acf.length} of ${info.max_lags} lags are flagged (outside the ${bandPercent} band and at least ${output.min_abs} in size).`
            : 'no lag is flagged.'
        }}
        Bars outside the dashed band are significant; flagged lags are drawn solid.
      </p>
      <p class="muted" data-role="lags">
        Significant ACF lags: {{ lagWords(info.significant_acf) }}. Significant PACF lags:
        {{ lagWords(info.significant_pacf) }}.
        <template v-if="info.strongest_lag !== null"
          >The strongest is lag {{ info.strongest_lag }}.</template
        >
      </p>
      <DsbfChart
        title="ACF"
        :theme="theme"
        :option="acfOption(info)"
        height="240px"
        :export-name="`${column}-acf`"
      />
      <DsbfChart
        title="PACF"
        :theme="theme"
        :option="pacfOption(info)"
        height="240px"
        :export-name="`${column}-pacf`"
      />
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useThemeStore } from '../../stores/theme'
import DsbfChart from '../charts/DsbfChart.vue'
import { acfOption, lagWords, pacfOption, type AutocorrelationOutput } from './timeseries'

const props = defineProps<{ output: AutocorrelationOutput; column: string }>()

const theme = computed(() => useThemeStore().theme)
const info = computed(() => props.output.columns[props.column])
const skipped = computed(() => props.output.skipped[props.column] ?? 'it is not a series column.')
const bandPercent = computed(() => `${Math.round((1 - props.output.alpha) * 100)}%`)
</script>
