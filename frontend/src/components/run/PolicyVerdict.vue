<template>
  <section class="card" aria-labelledby="policy-title">
    <h2 id="policy-title" class="card-title">Policy</h2>
    <p>
      <span :class="['badge', policy.failed ? 'badge-error' : 'badge-ok']">{{
        policy.failed ? 'failed' : 'passed'
      }}</span>
      <span class="muted"> {{ plural(policy.rules.length, 'rule') }} matched </span>
    </p>
    <ul>
      <li v-for="outcome in policy.rules" :key="outcome.rule.name">
        <strong>{{ outcome.rule.name }}</strong>
        <span class="muted">
          · {{ plural(outcome.finding_ids.length, 'finding') }},
          {{ plural(outcome.check_ids.length, 'check') }}
        </span>
        <span v-for="step in outcome.rule.then" :key="step.kind" class="badge badge-info step">{{
          step.kind
        }}</span>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import { plural } from '../../format'
import type { components } from '../../generated/api'

defineProps<{ policy: components['schemas']['PolicyOutcome'] }>()
</script>

<style scoped>
.step {
  margin-left: 6px;
}
</style>
