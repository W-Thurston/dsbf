import type { components } from '../../generated/api'
import { formatCount } from '../../format'

type Profile =
  | components['schemas']['NumericProfile']
  | components['schemas']['CategoricalProfile']
  | components['schemas']['BooleanProfile']
  | components['schemas']['DatetimeProfile']
  | components['schemas']['TextProfile']
  | components['schemas']['IdentifierProfile']
  | components['schemas']['UnknownProfile']

export type StatRow = [label: string, value: string]

function number(value: number | null | undefined): string | null {
  if (value === null || value === undefined) return null
  return Number.isInteger(value) ? formatCount(value) : String(Number(value.toPrecision(6)))
}

/** The profile's counts and its type's own statistics, as label and value, skipping absent ones. */
export function statRows(profile: Profile): StatRow[] {
  const { counts } = profile
  const rows: StatRow[] = [
    ['Rows', formatCount(counts.rows)],
    ['Null', formatCount(counts.nulls)],
    ['Distinct', `${formatCount(counts.distinct)}${counts.distinct_exact ? '' : ' (estimated)'}`],
    ['Stored type', profile.physical_dtype],
  ]
  const add = (label: string, value: string | null | undefined) => {
    if (value !== null && value !== undefined) rows.push([label, value])
  }
  const loose = profile as unknown as Record<string, unknown>
  const stats = loose.stats as Record<string, number | null> | null | undefined
  if (stats) {
    for (const key of ['min', 'max', 'mean', 'std', 'skewness', 'kurtosis']) {
      add(key, number(stats[key]))
    }
  }
  for (const key of ['zeros', 'non_finite', 'empty_strings', 'true_count', 'false_count']) {
    add(key.replace('_', ' '), number(loose[key] as number | undefined))
  }
  if (profile.semantic_type === 'datetime') {
    add('Earliest', profile.min)
    add('Latest', profile.max)
    add('Format', profile.format)
    add('Frequency', profile.inferred_frequency)
    add('Gaps', number(profile.n_gaps))
  }
  add('Entropy (bits)', number(loose.entropy_bits as number | null | undefined))
  const lengths = loose.lengths as { min: number; max: number; mean: number } | null | undefined
  if (lengths) add('Length', `${lengths.min}–${lengths.max}, mean ${number(lengths.mean)}`)
  return rows
}
