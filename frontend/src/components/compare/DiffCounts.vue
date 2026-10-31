<template>
  <section class="card" data-section="diff-counts" aria-labelledby="diff-counts-title">
    <h2 id="diff-counts-title" class="card-title">What changed</h2>
    <BarList :items="items" label="Changes by kind" />
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { components } from '../../generated/api'
import { diffCounts } from './diffCounts'
import BarList from '../overview/BarList.vue'

const props = defineProps<{ diff: components['schemas']['RunDiff'] }>()

const items = computed(() => diffCounts(props.diff))
</script>
