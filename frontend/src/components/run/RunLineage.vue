<template>
  <p v-if="entry" class="muted lineage" data-role="lineage">
    <template v-if="source.location"
      >Source <code>{{ source.location }}</code></template
    ><template v-else>In-memory source</template
    ><template v-if="source.format"> · {{ source.format }}</template>
    <template v-if="entry.previous_run">
      ·
      <span :class="['badge', changed ? 'badge-warn' : 'badge-ok']" data-role="content-change">{{
        changed
          ? 'content changed since the previous run'
          : 'content unchanged since the previous run'
      }}</span>
      <template v-if="entry.location_changed">
        <span class="badge badge-warn">location changed</span></template
      >
    </template>
  </p>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, unwrap } from '../../api/client'
import type { components } from '../../generated/api'

const props = defineProps<{
  runId: string
  datasetKey: string
  source: components['schemas']['SourceRef']
}>()

interface Entry {
  previous_run: boolean
  location_changed: boolean
  fingerprint_changed: boolean
}
const entry = ref<Entry | null>(null)
const changed = computed(() => entry.value?.fingerprint_changed ?? false)

watch(
  () => props.runId,
  async () => {
    entry.value = { previous_run: false, location_changed: false, fingerprint_changed: false }
    if (!props.datasetKey) return
    try {
      const entries = await unwrap(
        api.GET('/v2/datasets/{key}/lineage', {
          params: { path: { key: props.datasetKey }, query: { limit: 500 } },
        }),
      )
      const items = entries.items
      const at = items.findIndex((e) => e.run_id === props.runId)
      const here = items[at]
      entry.value = {
        previous_run: at >= 0 && at < items.length - 1,
        location_changed: here?.location_changed ?? false,
        fingerprint_changed: here?.fingerprint_changed ?? false,
      }
    } catch {
      // The header then shows the source alone.
    }
  },
  { immediate: true },
)
</script>

<style scoped>
.lineage {
  margin: 2px 0 0;
}
</style>
