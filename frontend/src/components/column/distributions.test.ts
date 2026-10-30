import { describe, expect, it } from 'vitest'
import { columnBoxOption, distributionOption } from './chartOptions'
import {
  bimodalText,
  browserGroups,
  componentCurves,
  normalitySummary,
  outlierConsensus,
  outlierRows,
  statStrip,
  topCorrelations,
  warnedColumns,
  type Profile,
} from './distributions'

const column = (name: string, semantic_type: string, nulls = 0) =>
  ({
    name,
    semantic_type,
    physical_dtype: 'x',
    position: 0,
    counts: { rows: 100, nulls, distinct: 5, distinct_exact: true },
    sketch: {
      bins: [],
      min: null,
      max: null,
      mode: null,
      mode_share: null,
      skewness: null,
      top_values: [],
    },
  }) as never

describe('column browser', () => {
  const columns = [
    column('b', 'categorical', 50),
    column('a', 'numeric', 10),
    column('c', 'numeric'),
  ]

  it('groups by type in a fixed order and carries null share and warning', () => {
    const groups = browserGroups(columns, new Set(['c']), '')
    expect(groups.map((g) => g.type)).toEqual(['numeric', 'categorical'])
    expect(groups[0]!.columns).toEqual([
      { name: 'a', nullShare: 0.1, warned: false },
      { name: 'c', nullShare: 0, warned: true },
    ])
  })

  it('searches names without regard to case', () => {
    expect(
      browserGroups(columns, new Set(), 'B').flatMap((g) => g.columns.map((c) => c.name)),
    ).toEqual(['b'])
  })

  it('warns only for warn or worse findings', () => {
    const subject = (c: string) => ({ columns: [c] })
    expect(
      warnedColumns([
        { severity: 'notice', subject: subject('a') },
        { severity: 'warn', subject: subject('b') },
        { severity: 'error', subject: subject('c') },
      ]),
    ).toEqual(new Set(['b', 'c']))
  })
})

describe('stats strip', () => {
  it('reads a numeric column and a categorical one', () => {
    const numeric = {
      semantic_type: 'numeric',
      counts: { rows: 10, nulls: 2, distinct: 8, distinct_exact: true },
      stats: {
        mean: 4,
        std: 1,
        min: 0,
        max: 9,
        skewness: 0.5,
        kurtosis: 3,
        quantiles: [
          { p: 0.25, value: 2 },
          { p: 0.5, value: 4 },
          { p: 0.75, value: 6 },
        ],
      },
    } as unknown as Profile
    const strip = Object.fromEntries(statStrip(numeric, 3).map((s) => [s.label, s.value]))
    expect(strip).toMatchObject({ Count: '8', Median: '4', IQR: '4', Null: '20.0%', Outliers: '3' })

    const categorical = {
      semantic_type: 'categorical',
      counts: { rows: 10, nulls: 0, distinct: 2, distinct_exact: true },
      top_values: [{ value: 'male', count: 6 }],
    } as unknown as Profile
    const cat = Object.fromEntries(statStrip(categorical, null).map((s) => [s.label, s.value]))
    expect(cat).toMatchObject({ Unique: '2', 'Top value': 'male', 'Top share': '60.0%' })
  })
})

describe('outliers and normality', () => {
  const outlier = {
    n_used: 10,
    iqr: { multiplier: 1.5, count: 2, share: 0.2, q1: 1, q3: 3, lower_fence: -2, upper_fence: 6 },
    zscore: { count: 1, share: 0.1, threshold: 3, max_score: 4 },
    mad: null,
    methods_flagging: ['iqr', 'zscore'],
    consensus: true,
  }

  it('lists the methods that ran and says whether they agree', () => {
    expect(outlierRows(outlier).map((r) => [r.method, r.count])).toEqual([
      ['IQR', 2],
      ['Z-score', 1],
    ])
    expect(outlierConsensus(outlier)).toMatch(/2 methods/)
    expect(outlierConsensus({ ...outlier, consensus: false, methods_flagging: [] })).toMatch(
      /No method/,
    )
  })

  it('reports the verdict and tiny p-values', () => {
    const test = (p: number, rejects: boolean) => ({
      test: 'shapiro_wilk',
      statistic: 1,
      p_value: p,
      rejects,
    })
    const summary = normalitySummary({
      n_used: 5,
      primary: test(1e-8, true),
      jarque_bera: test(0.2, false),
      non_normal: true,
    })
    expect(summary.verdict).toBe('Not normal')
    expect(summary.tests.map((t) => t.p)).toEqual(['< 0.001', '0.200'])
  })
})

describe('bimodal components', () => {
  const bimodal = {
    n_finite: 100,
    bic_delta: 50,
    separation: 4,
    means: [2, 7] as [number, number],
    stds: [1, 1] as [number, number],
    weights: [0.5, 0.5] as [number, number],
    bimodal: true,
  }
  const edges = [0, 2, 4, 6, 8, 10]
  const counts = [20, 10, 0, 10, 20]

  it('scales each component to counts per bin and peaks at its mean', () => {
    const [low, high] = componentCurves(bimodal, edges, counts)
    expect(low!.map(([x]) => x)).toEqual([1, 3, 5, 7, 9])
    const peak = low!.reduce((a, b) => (b[1] > a[1] ? b : a))
    expect(peak[0]).toBe(1)
    expect(high!.reduce((a, b) => (b[1] > a[1] ? b : a))[0]).toBe(7)
    // a normal's mass is one, so a component holds half of the 60 values in this bin width
    const mass = low!.reduce((n, [, y]) => n + y, 0)
    expect(mass).toBeGreaterThan(20)
    expect(mass).toBeLessThan(31)
  })

  it('adds the components as line series only when given', () => {
    const histogram = { edges, counts, rule: 'fd' as const }
    const marks = { mean: 5, median: 5 }
    const plain = distributionOption(histogram, 'dark', marks)
    const over = distributionOption(
      histogram,
      'dark',
      marks,
      componentCurves(bimodal, edges, counts),
    )
    expect((over.series as unknown[]).length).toBe((plain.series as unknown[]).length + 2)
  })

  it('describes the peaks', () => {
    expect(bimodalText(bimodal)).toMatch(/around 2 \(50\.0% of values\) and 7/)
  })
})

describe('box plot and correlations', () => {
  it('clips whiskers to the fences', () => {
    const option = columnBoxOption(
      { min: 0, q1: 2, median: 3, q3: 4, max: 100 },
      { lower: -1, upper: 7 },
      'dark',
    )
    const [series] = option.series as { data: number[][] }[]
    expect(series!.data[0]).toEqual([0, 2, 3, 4, 7])
  })

  it("ranks a column's associations by strength, either way round", () => {
    const cell = (first: string, second: string, value: number) =>
      ({ first, second, value, statistic: 'pearson_r' }) as never
    const top = topCorrelations(
      [cell('a', 'b', 0.2), cell('c', 'a', -0.9), cell('b', 'c', 0.99), cell('a', 'd', 0.5)],
      'a',
      2,
    )
    expect(top.map((t) => t.other)).toEqual(['c', 'd'])
  })
})
