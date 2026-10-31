<template>
  <section class="card" aria-labelledby="stationarity-title" data-section="stationarity">
    <h2 id="stationarity-title" class="card-title">Stationarity</h2>
    <p class="muted">
      ADF has a unit root as its null; KPSS has stationarity. Each is also run with a linear trend.
      Tests at {{ Math.round(output.alpha * 100) }}%.
    </p>
    <details class="reading" data-role="reading">
      <summary>How to read the two tests together</summary>
      <ul>
        <li>ADF rejects and KPSS does not: stationary.</li>
        <li>KPSS rejects and ADF does not: not stationary, it has a unit root.</li>
        <li>
          Both reject: the series has a trend or a structural break; the trend variants say which.
        </li>
        <li>Neither rejects: inconclusive; the series may be too short to tell.</li>
      </ul>
    </details>
    <table class="data-table">
      <thead>
        <tr>
          <th scope="col">Column</th>
          <th scope="col">Assessment</th>
          <th scope="col">ADF p</th>
          <th scope="col">KPSS p</th>
          <th scope="col">ADF p (trend)</th>
          <th scope="col">KPSS p (trend)</th>
          <th scope="col">Confidence</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(info, name) in output.columns"
          :key="name"
          :data-column="name"
          :class="{ selected: name === column }"
        >
          <th scope="row">
            <button class="link" type="button" @click="$emit('select', String(name))">
              {{ name }}
            </button>
          </th>
          <td>
            <span :class="['badge', assessmentClass(info.assessment)]" data-role="assessment">{{
              assessmentWords(info.assessment)
            }}</span>
          </td>
          <td>{{ pValueWords(info.adf) }}</td>
          <td>{{ pValueWords(info.kpss) }}</td>
          <td>{{ pValueWords(info.adf_trend) }}</td>
          <td>{{ pValueWords(info.kpss_trend) }}</td>
          <td>{{ info.confidence }}</td>
        </tr>
      </tbody>
    </table>
  </section>
</template>

<script setup lang="ts">
import {
  assessmentClass,
  assessmentWords,
  pValueWords,
  type StationarityOutput,
} from './timeseries'

defineProps<{ output: StationarityOutput; column: string }>()
defineEmits<{ select: [column: string] }>()
</script>

<style scoped>
td {
  white-space: nowrap;
}
.reading {
  margin: 8px 0;
}
.link {
  background: none;
  border: 0;
  padding: 0;
  color: var(--link);
  font: inherit;
  cursor: pointer;
  text-decoration: underline;
}
.selected {
  background: var(--bg);
}
</style>
