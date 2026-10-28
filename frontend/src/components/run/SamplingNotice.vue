<template>
  <p class="notice" role="note">
    Profiled a {{ sampling.strategy }} sample of {{ formatCount(sampling.sampled_rows) }} of
    {{ formatCount(sampling.original_rows) }} rows<template v-if="sampling.seed !== null">
      (seed {{ sampling.seed }})</template
    ><template v-if="sampling.stratify_by.length">
      stratified by {{ sampling.stratify_by.join(', ') }}</template
    >. Counts and estimates describe the sample, not the whole dataset.
  </p>
</template>

<script setup lang="ts">
import { formatCount } from '../../format'
import type { components } from '../../generated/api'

defineProps<{ sampling: components['schemas']['SamplingInfo'] }>()
</script>

<style scoped>
.notice {
  margin: 0;
  padding: 10px 14px;
  border-radius: 8px;
  background: var(--notice-bg);
  color: var(--notice-fg);
}
</style>
