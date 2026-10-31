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

<style scoped>
.rerun {
  margin-top: 16px;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--level-blue);
  border-radius: 8px;
  background: var(--bg);
  font-size: 14px;
}
.rerun[data-state='succeeded'] {
  border-left-color: var(--level-green);
}
.rerun[data-state='failed'],
.rerun[data-state='cancelled'],
.rerun[data-state='interrupted'] {
  border-left-color: var(--level-red);
}
.rerun p {
  margin: 0;
}
[data-role='job-state'] {
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-size: 12px;
}
.error {
  margin-top: 10px;
}
</style>
