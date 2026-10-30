<template>
  <section class="card" aria-labelledby="types-title" data-section="column-types">
    <h2 id="types-title" class="card-title">Column types</h2>
    <BarList :items="items" label="Columns per type" />
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { components } from '../../generated/api'
import BarList from './BarList.vue'
import { typeCounts } from './overview'

const props = defineProps<{ columns: components['schemas']['ColumnSummary'][] }>()
const items = computed(() => typeCounts(props.columns))
</script>
