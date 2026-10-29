<template>
  <div class="rerun" data-role="rerun-progress" :data-state="job.state">
    <p>
      Rerun <strong data-role="job-state">{{ job.state }}</strong>
      <template v-if="last"> — {{ describe(last) }}</template>
    </p>
    <p v-if="job.error" class="error" data-role="job-error">{{ job.error }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { components } from '../../generated/api'

type Job = components['schemas']['Job']
type JobEvent = components['schemas']['JobEvent']

const props = defineProps<{ job: Job; events: JobEvent[] }>()

const last = computed(() => props.events.filter((event) => event.kind !== 'job_finished').at(-1))

function describe(event: JobEvent): string {
  const where = event.position && event.total ? ` (${event.position} of ${event.total})` : ''
  return `${event.check_id ?? 'working'}${where}`
}
</script>
