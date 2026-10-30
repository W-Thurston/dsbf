import { ref, watch } from 'vue'
import { api, unwrap } from '../api/client'
import type { components } from '../generated/api'
import type {
  BimodalColumn,
  NormalityColumn,
  OutlierColumn,
} from '../components/column/distributions'
import type { QQData } from '../components/column/chartOptions'

type Columns = components['schemas']['ColumnSummary'][]
type Cells = components['schemas']['RelationshipCell'][]
type Findings = { severity: components['schemas']['Severity']; subject: { columns: string[] } }[]

export interface PerColumn<T> {
  columns: Record<string, T>
  skipped: Record<string, string>
}
export type NormalityEntry = NormalityColumn & { qq?: QQData | null }

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

const WARNED_PAGE = 100

/** Every finding of warn or worse, paged through. */
async function warnedFindings(runId: string) {
  const items: Findings = []
  try {
    for (let offset = 0; ; offset += WARNED_PAGE) {
      const page = await unwrap(
        api.GET('/v2/runs/{run_id}/findings', {
          params: {
            path: { run_id: runId },
            query: { min_severity: 'warn', limit: WARNED_PAGE, offset },
          },
        }),
      )
      items.push(...page.items)
      if (offset + WARNED_PAGE >= page.total) break
    }
  } catch {
    // a missing dot is not worth failing the tab
  }
  return { items }
}

/** What the Distributions tab shares across its columns, fetched once per run. */
export function useDistributionsData(runId: () => string) {
  const columns = ref<Columns | null>(null)
  const warnings = ref<Findings>([])
  const outliers = ref<PerColumn<OutlierColumn> | null>(null)
  const normality = ref<PerColumn<NormalityEntry> | null>(null)
  const bimodality = ref<PerColumn<BimodalColumn> | null>(null)
  const cells = ref<Cells>([])
  const error = ref('')

  watch(
    runId,
    async (run) => {
      columns.value = outliers.value = normality.value = bimodality.value = null
      warnings.value = []
      cells.value = []
      error.value = ''
      const path = { params: { path: { run_id: run } } }
      try {
        columns.value = await unwrap(api.GET('/v2/runs/{run_id}/columns', path))
      } catch (caught) {
        error.value = caught instanceof Error ? caught.message : String(caught)
        return
      }
      const [flagged, out, norm, bi, matrix] = await Promise.all([
        warnedFindings(run),
        checkOutput<PerColumn<OutlierColumn>>(run, 'outliers'),
        checkOutput<PerColumn<NormalityEntry>>(run, 'normality'),
        checkOutput<PerColumn<BimodalColumn>>(run, 'bimodality'),
        unwrap(api.GET('/v2/runs/{run_id}/relationships', path)).catch(() => null),
      ])
      warnings.value = flagged?.items ?? []
      outliers.value = out
      normality.value = norm
      bimodality.value = bi
      cells.value = matrix?.cells ?? []
    },
    { immediate: true },
  )

  return { columns, warnings, outliers, normality, bimodality, cells, error }
}
