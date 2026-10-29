// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ColumnSample from './ColumnSample.vue'
import { histogramOption, qqOf, qqOption, topValuesOption } from './chartOptions'
import { statRows } from './columnStats'

const get = vi.hoisted(() => vi.fn())
vi.mock('../../api/client', async (original) => ({
  ...(await original<typeof import('../../api/client')>()),
  api: { GET: get },
}))

afterEach(() => get.mockReset())

const QQ = {
  theoretical: [-1, 0, 1],
  empirical: [-2, 0, 3],
  reference_x: [-0.67, 0.67] as [number, number],
  reference_y: [-1, 1] as [number, number],
}

describe('chart options', () => {
  it('draws each histogram bin between its stored edges', () => {
    const option = histogramOption({ edges: [0, 10, 20], counts: [3, 5], rule: 'fd' })
    const [series] = option.series as { data: number[][] }[]
    expect(series!.data).toEqual([
      [0, 10, 3],
      [10, 20, 5],
    ])
    expect(option.xAxis).toMatchObject({ type: 'value', min: 0, max: 20 })
  })

  it('has no histogram option for a column without one', () => {
    expect(histogramOption(null)).toEqual({})
  })

  it('lists top values in the order given', () => {
    const option = topValuesOption([
      { value: 'a', count: 5 },
      { value: 2, count: 1 },
    ])
    expect(option.yAxis).toMatchObject({ data: ['a', '2'] })
    expect((option.series as { data: number[] }[])[0]!.data).toEqual([5, 1])
  })

  it('pairs Q-Q quantiles and draws the reference line from the stored points', () => {
    const [points, line] = qqOption(QQ).series as { data: number[][] }[]
    expect(points!.data).toEqual([
      [-1, -2],
      [0, 0],
      [1, 3],
    ])
    expect(line!.data).toEqual([
      [-0.67, -1],
      [0.67, 1],
    ])
  })
})

describe('qqOf', () => {
  const output = {
    columns: { Fare: { qq: QQ }, Age: { qq: null } },
    skipped: { Name: 'not numeric' },
  }

  it('returns the stored quantiles of a flagged column', () => {
    expect(qqOf(output, 'Fare')).toEqual({ qq: QQ })
  })

  it('says why a column has none', () => {
    expect(qqOf(output, 'Age')).toMatchObject({ reason: expect.stringContaining('passed') })
    expect(qqOf(output, 'Name')).toMatchObject({ reason: expect.stringContaining('not numeric') })
    expect(qqOf(output, 'Other')).toMatchObject({ reason: expect.stringContaining('did not test') })
    expect(qqOf(null, 'Age')).toMatchObject({ reason: expect.stringContaining('did not test') })
  })
})

describe('statRows', () => {
  it('shows the counts and a numeric profile statistics, skipping absent ones', () => {
    const rows = statRows({
      semantic_type: 'numeric',
      physical_dtype: 'Float64',
      counts: { rows: 1000, nulls: 20, distinct: 80, distinct_exact: false },
      zeros: 0,
      non_finite: 0,
      stats: { min: 0.42, max: 80, mean: 29.7, std: null },
    } as never)
    expect(Object.fromEntries(rows)).toMatchObject({
      Rows: '1,000',
      Distinct: '80 (estimated)',
      min: '0.42',
      mean: '29.7',
      zeros: '0',
    })
    expect(rows.map(([label]) => label)).not.toContain('std')
  })
})

describe('ColumnSample', () => {
  it('shows only the column, numbered from the page offset', async () => {
    get.mockResolvedValue({
      data: {
        columns: ['id', 'Age'],
        rows: [
          [1, 22],
          [2, null],
        ],
        total: 2,
        offset: 0,
        limit: 20,
        sampling: null,
      },
      response: new Response(),
    })
    const sample = mount(ColumnSample, { props: { runId: 'r', column: 'Age' } })
    await flushPromises()

    const cells = sample.findAll('tbody td').map((td) => td.text())
    expect(cells).toEqual(['1', '22', '2', 'null'])
  })

  it('says so when the run kept no snapshot', async () => {
    get.mockResolvedValue({
      error: { detail: 'run kept no data snapshot' },
      response: new Response(null, { status: 404 }),
    })
    const sample = mount(ColumnSample, { props: { runId: 'r', column: 'Age' } })
    await flushPromises()

    expect(sample.text()).toContain('kept no data snapshot')
    expect(sample.find('table').exists()).toBe(false)
  })
})
