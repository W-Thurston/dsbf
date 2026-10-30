import {
  computed,
  inject,
  provide,
  ref,
  watch,
  type ComputedRef,
  type InjectionKey,
  type Ref,
} from 'vue'
import { api, unwrap } from '../api/client'
import type { components } from '../generated/api'
import type { QualityOutput } from '../components/quality/quality'

type RunOverview = components['schemas']['RunOverview']

/** What every page of one run shares: its overview and its quality dimensions. */
export interface RunState {
  runId: ComputedRef<string>
  overview: Ref<RunOverview | null>
  /** None when the run lacks the quality_dimensions check or it did not finish. */
  dimensions: Ref<QualityOutput | null>
  error: Ref<string>
}

const KEY: InjectionKey<RunState> = Symbol('run')

/** Load a run once for the shell and every tab beneath it. */
export function provideRun(runId: () => string): RunState {
  const id = computed(runId)
  const overview = ref<RunOverview | null>(null)
  const dimensions = ref<QualityOutput | null>(null)
  const error = ref('')

  watch(
    id,
    async (run) => {
      overview.value = null
      dimensions.value = null
      error.value = ''
      try {
        overview.value = await unwrap(
          api.GET('/v2/runs/{run_id}', { params: { path: { run_id: run } } }),
        )
      } catch (caught) {
        error.value = caught instanceof Error ? caught.message : String(caught)
        return
      }
      try {
        const result = await unwrap(
          api.GET('/v2/runs/{run_id}/checks/{check_id}', {
            params: { path: { run_id: run, check_id: 'quality_dimensions' } },
          }),
        )
        if (result.status === 'ok') dimensions.value = result.output as unknown as QualityOutput
      } catch {
        // The health bar then says every dimension is unassessed.
      }
    },
    { immediate: true },
  )

  const state = { runId: id, overview, dimensions, error }
  provide(KEY, state)
  return state
}

export function useRun(): RunState {
  const state = inject(KEY)
  if (!state) throw new Error('useRun needs the run shell above it')
  return state
}
