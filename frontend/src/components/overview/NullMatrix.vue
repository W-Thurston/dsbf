<template>
  <div class="scroll">
    <table class="matrix" :aria-label="label" data-role="null-matrix">
      <thead>
        <tr>
          <td />
          <th v-for="name in labels" :key="name" scope="col" class="col">
            <span>{{ name }}</span>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, y) in labels" :key="row">
          <th scope="row">{{ row }}</th>
          <td
            v-for="(col, x) in labels"
            :key="col"
            :style="{ '--share': values[y]?.[x] ?? 0 }"
            :title="`${row} and ${col}: ${Math.round((values[y]?.[x] ?? 0) * 100)}%`"
          >
            {{ (values[y]?.[x] ?? 0) > 0 ? Math.round(values[y]![x]! * 100) : '' }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
defineProps<{ labels: string[]; values: number[][]; label: string }>()
</script>

<style scoped>
.scroll {
  overflow-x: auto;
}
.matrix {
  border-collapse: separate;
  border-spacing: 2px;
  font-size: 12px;
}
th {
  font-weight: 500;
  text-align: right;
  padding-right: 8px;
  white-space: nowrap;
}
.col {
  text-align: left;
  vertical-align: bottom;
  padding: 0 0 4px;
  height: 90px;
}
.col span {
  display: inline-block;
  writing-mode: vertical-rl;
  transform: rotate(180deg);
}
td {
  width: 38px;
  height: 30px;
  text-align: center;
  border-radius: 4px;
  font-variant-numeric: tabular-nums;
  /* The share sets how much of the primary colour shows over the card. */
  background: color-mix(in srgb, var(--chart-primary) calc(var(--share) * 100%), var(--border));
  color: var(--text);
}
</style>
