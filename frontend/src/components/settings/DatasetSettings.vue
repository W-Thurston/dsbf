<template>
  <section class="card" aria-labelledby="settings-title" data-section="settings">
    <h2 id="settings-title" class="card-title">Settings</h2>
    <p v-if="loadError" class="error">{{ loadError }}</p>
    <p v-else-if="fields === null || overlay === null" class="loading">Loading…</p>
    <template v-else>
      <p class="muted">
        Saved for this dataset and used by every rerun. Fields left at their default are not saved.
      </p>
      <SettingsFields v-model="overlay" :fields="fields" />
      <p v-if="saveError" class="error" data-role="save-error">{{ saveError }}</p>
      <div class="actions">
        <button type="button" data-role="save" :disabled="busy" @click="save">Save</button>
        <button type="button" data-role="reset" :disabled="busy" @click="reset">Reset</button>
        <button type="button" data-role="rerun" :disabled="busy || running" @click="rerun">
          Save and rerun
        </button>
      </div>
      <RerunProgress v-if="job" :job="job" :events="events" />
    </template>
  </section>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api, unwrap } from '../../api/client'
import type { components } from '../../generated/api'
import {
  fieldsOf,
  pruned,
  type Field,
  type JsonSchema,
  type Overlay,
} from '../../composables/schemaForm'
import RerunProgress from './RerunProgress.vue'
import SettingsFields from './SettingsFields.vue'

type Job = components['schemas']['Job']
type JobEvent = components['schemas']['JobEvent']

const props = defineProps<{ datasetKey: string }>()
const router = useRouter()

const POLL_MS = 400

const fields = ref<Field[] | null>(null)
const overlay = ref<Overlay | null>(null)
const loadError = ref('')
const saveError = ref('')
const busy = ref(false)
const job = ref<Job | null>(null)
const events = ref<JobEvent[]>([])
const running = ref(false)
let timer: ReturnType<typeof setTimeout> | undefined
let alive = true

const path = () => ({ params: { path: { key: props.datasetKey } } })
const message = (caught: unknown) => (caught instanceof Error ? caught.message : String(caught))

onMounted(async () => {
  try {
    const document = await (await fetch('/v2/openapi.json')).json()
    const schemas = document.components.schemas as Record<string, JsonSchema>
    // The request body's `settings` is the ProfileConfig the job would run with.
    fields.value = fieldsOf(schemas['ProfileConfig']!, schemas)
    overlay.value = (await unwrap(api.GET('/v2/datasets/{key}/settings', path()))) as Overlay
  } catch (caught) {
    loadError.value = message(caught)
  }
})
onUnmounted(() => {
  alive = false
  clearTimeout(timer)
})

async function save(): Promise<boolean> {
  saveError.value = ''
  busy.value = true
  try {
    const body = pruned(overlay.value ?? {})
    overlay.value = (await unwrap(
      api.PUT('/v2/datasets/{key}/settings', { ...path(), body }),
    )) as Overlay
    return true
  } catch (caught) {
    saveError.value = message(caught)
    return false
  } finally {
    busy.value = false
  }
}

async function reset() {
  saveError.value = ''
  busy.value = true
  try {
    const { response } = await api.DELETE('/v2/datasets/{key}/settings', path())
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
    overlay.value = {}
  } catch (caught) {
    saveError.value = message(caught)
  } finally {
    busy.value = false
  }
}

async function rerun() {
  if (!(await save())) return
  try {
    job.value = await unwrap(api.POST('/v2/datasets/{key}/rerun', path()))
    events.value = []
    running.value = true
    poll(0)
  } catch (caught) {
    saveError.value = message(caught)
  }
}

async function poll(after: number) {
  if (!alive || !job.value) return
  try {
    const progress = await unwrap(
      api.GET('/v2/jobs/{job_id}', {
        params: { path: { job_id: job.value.job_id }, query: { after } },
      }),
    )
    job.value = progress.job
    events.value = [...events.value, ...progress.events]
    if (progress.job.state === 'succeeded' && progress.job.run_id) {
      running.value = false
      await router.push({ name: 'run', params: { runId: progress.job.run_id } })
      return
    }
    if (['failed', 'cancelled', 'interrupted'].includes(progress.job.state)) {
      running.value = false
      return
    }
    timer = setTimeout(() => poll(progress.next_after), POLL_MS)
  } catch (caught) {
    running.value = false
    saveError.value = message(caught)
  }
}
</script>

<style scoped>
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}
button {
  font-size: 14px;
  padding: 7px 14px;
  border-radius: 6px;
  color: var(--text);
  background: transparent;
  border: 1px solid var(--border);
}
button:hover:not(:disabled) {
  border-color: var(--link);
}
button:focus-visible {
  outline: 2px solid var(--link);
  outline-offset: 2px;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
button[data-role='rerun'] {
  margin-left: auto;
  color: var(--bg);
  background: var(--link);
  border-color: var(--link);
  font-weight: 600;
}
button[data-role='reset'] {
  color: var(--text-muted);
}
.muted {
  margin: 0 0 14px;
  font-size: 13px;
}
[data-role='save-error'] {
  margin: 12px 0 0;
}
</style>
