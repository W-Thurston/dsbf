// The transformation preview: for each skewed column, how far each candidate transform
// brings the skewness toward zero, with the recommended one marked.

export interface TransformCandidate {
  skewness_after: number | null
  lam: number | null
  skipped: string | null
}

export interface ColumnTransforms {
  skewness_before: number
  recommended: string | null
  candidates: Record<string, TransformCandidate>
}

export interface TransformationsOutput {
  columns: Record<string, ColumnTransforms>
  skipped_columns: Record<string, string>
}

export interface SkewBar {
  label: string
  count: number
  text: string
}

const signed = (value: number) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`

/** Skewness before and after each candidate as bars of its size; a skipped one has none. */
export function skewBars(column: ColumnTransforms): { bars: SkewBar[]; skipped: string[] } {
  const bars: SkewBar[] = [
    {
      label: 'before',
      count: Math.abs(column.skewness_before),
      text: signed(column.skewness_before),
    },
  ]
  const skipped: string[] = []
  for (const [name, candidate] of Object.entries(column.candidates)) {
    if (candidate.skewness_after === null) {
      skipped.push(`${name}: ${candidate.skipped ?? 'not applied'}`)
      continue
    }
    bars.push({
      label: name === column.recommended ? `${name} ★` : name,
      count: Math.abs(candidate.skewness_after),
      text: signed(candidate.skewness_after),
    })
  }
  return { bars, skipped }
}
