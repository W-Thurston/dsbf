import { ref, watch } from 'vue'
import { api, unwrap } from '../api/client'
import type { components } from '../generated/api'

type Columns = components['schemas']['ColumnSummary'][]
type Facts = components['schemas']['DatasetFacts']

export interface DuplicateRows {
  duplicate_rows: number
  duplicate_share: number
}
export interface SampleSize {
  verdict: 'adequate' | 'limited' | 'insufficient'
  rows_per_feature: number
}
export interface DtypeSuggestion {
  current_dtype: string
  suggested_dtype: string
  estimated_saving_bytes: number
}
export interface DtypeOptimizations {
  suggestions: Record<string, DtypeSuggestion>
  total_saving_bytes: number
}

/** The output of a check that finished, or null when it is absent or did not. */
async function checkOutput<T>(runId: string, checkId: string): Promise<T | null> {
  try {
    const result = await unwrap(
      api.GET('/v2/runs/{run_id}/checks/{check_id}', {
        params: { path: { run_id: runId, check_id: checkId } },
      }),
    )
    return result.status === 'ok' ? (result.output as unknown as T) : null
  } catch {
    return null
  }
}

/** Everything the Overview draws beyond the run's overview, fetched in parallel. */
export function useOverviewData(runId: () => string) {
  const columns = ref<Columns | null>(null)
  const facts = ref<Facts | null>(null)
  const duplicates = ref<DuplicateRows | null>(null)
  const sampleSize = ref<SampleSize | null>(null)
  const dtypes = ref<DtypeOptimizations | null>(null)
  const error = ref('')

  watch(
    runId,
    async (run) => {
      columns.value = facts.value = duplicates.value = sampleSize.value = dtypes.value = null
      error.value = ''
      try {
        const params = { params: { path: { run_id: run } } }
        const [cols, dataset] = await Promise.all([
          unwrap(api.GET('/v2/runs/{run_id}/columns', params)),
          unwrap(api.GET('/v2/runs/{run_id}/dataset', params)),
        ])
        columns.value = cols
        facts.value = dataset
      } catch (caught) {
        error.value = caught instanceof Error ? caught.message : String(caught)
        return
      }
      // A check that did not finish leaves its figure out of the strip.
      ;[duplicates.value, sampleSize.value, dtypes.value] = await Promise.all([
        checkOutput<DuplicateRows>(run, 'duplicate_rows'),
        checkOutput<SampleSize>(run, 'sample_size'),
        checkOutput<DtypeOptimizations>(run, 'dtype_optimizations'),
      ])
    },
    { immediate: true },
  )

  return { columns, facts, duplicates, sampleSize, dtypes, error }
}
