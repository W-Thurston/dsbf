// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import HealthBar from '../shell/HealthBar.vue'
import LevelBanner from './LevelBanner.vue'
import Sparkline from './SparklineChart.vue'
import StrengthBar from './StrengthBar.vue'
import TrafficCard from './TrafficCard.vue'
import { binIndex, boxPlot, heatmap, histogramBars, topValues } from './options'
import { MARKS } from '../../chartTheme'

type Series = Record<string, unknown>
const seriesOf = (option: { series?: unknown }) => option.series as Series[]

describe('histogramBars', () => {
  const edges = [0, 10, 20, 30]

  it('draws one bar per bin on the stored edges', () => {
    const [bars] = seriesOf(histogramBars(edges, [3, 5, 1], 'dark'))
    expect(bars?.data).toEqual([
      [0, 10, 3],
      [10, 20, 5],
      [20, 30, 1],
    ])
  })

  it('draws a reference line at its value in the theme colour', () => {
    const line = seriesOf(
      histogramBars(edges, [3, 5, 1], 'light', [{ value: 12, label: 'mean' }]),
    )[1]
    const mark = line?.markLine as { data: unknown[]; lineStyle: { color: string } }
    expect(mark.data).toEqual([{ xAxis: 12, name: 'mean' }])
    expect(mark.lineStyle.color).toBe(MARKS.light.reference)
  })

  it('finds the bin that holds a value, clamping outside the range', () => {
    expect([
      binIndex(edges, 5),
      binIndex(edges, 10),
      binIndex(edges, 99),
      binIndex(edges, -4),
    ]).toEqual([0, 1, 2, 0])
  })
})

describe('the other chart options', () => {
  it('draws a box per row from its five numbers', () => {
    const [box] = seriesOf(
      boxPlot([{ label: 'age', min: 0, q1: 1, median: 2, q3: 3, max: 4 }], 'dark'),
    )
    expect(box?.data).toEqual([[0, 1, 2, 3, 4]])
  })

  it('lists the top values in the order given', () => {
    const option = topValues(
      [
        { label: 'a', count: 3 },
        { label: 'b', count: 1 },
      ],
      'dark',
    )
    expect((option.yAxis as { data: string[] }).data).toEqual(['a', 'b'])
    expect(seriesOf(option)[0]?.data).toEqual([3, 1])
  })

  it('centres a signed heatmap on zero', () => {
    const option = heatmap(['a'], ['a'], [{ x: 0, y: 0, value: 1 }], 'dark', {
      min: -1,
      max: 1,
      signed: true,
    })
    const colors = (option.visualMap as { inRange: { color: string[] } }).inRange.color
    expect(colors).toHaveLength(3)
  })
})

describe('Sparkline', () => {
  it('draws a line through finite values and nothing for fewer than two', () => {
    expect(
      mount(Sparkline, { props: { values: [1, 3, 2] } })
        .find('polyline')
        .exists(),
    ).toBe(true)
    expect(
      mount(Sparkline, { props: { values: [1, Number.NaN] } })
        .find('polyline')
        .exists(),
    ).toBe(false)
  })

  it('draws a constant series flat', () => {
    const points = mount(Sparkline, { props: { values: [4, 4, 4], height: 20 } })
      .find('polyline')
      .attributes('points')!
    expect(new Set(points.split(' ').map((p) => p.split(',')[1]))).toEqual(new Set(['10.0']))
  })
})

describe('StrengthBar', () => {
  it('sizes segments by value, omits empty ones and describes itself', () => {
    const bar = mount(StrengthBar, {
      props: {
        segments: [
          { label: 'strong', value: 3, mark: 'red' },
          { label: 'weak', value: 0, mark: 'green' },
          { label: 'moderate', value: 1, mark: 'amber' },
        ],
      },
    })
    expect(bar.findAll('.seg')).toHaveLength(2)
    expect(bar.attributes('aria-label')).toBe('strong 3, weak 0, moderate 1')
  })
})

describe('TrafficCard and LevelBanner', () => {
  it('say the level in words as well as colour', () => {
    const card = mount(TrafficCard, {
      props: { title: 'Completeness', mark: 'none', summary: 'Not assessed' },
    })
    expect(card.find('[role="img"]').attributes('aria-label')).toBe('not assessed')
    expect(card.text()).toContain('Not assessed')

    const banner = mount(LevelBanner, {
      props: { mark: 'red', headline: 'Serious', detail: '5 columns' },
    })
    expect(banner.find('[role="img"]').attributes('aria-label')).toBe('serious')
    expect(banner.text()).toContain('5 columns')
  })
})

describe('HealthBar', () => {
  const dimension = (over: object) => ({
    assessed: true,
    level: 'green',
    coverage: 1,
    inputs: [],
    affected_columns: [],
    affected_count: 0,
    pct_affected: 0,
    findings: [],
    ...over,
  })

  it('shows a dot and a summary per dimension, grey for the unassessed', () => {
    const bar = mount(HealthBar, {
      props: {
        tab: 'overview',
        dimensions: {
          total_columns: 5,
          clean_columns: [],
          dimensions: {
            completeness: dimension({ level: 'red', affected_count: 2, pct_affected: 0.4 }),
            validity: dimension({}),
            usability: dimension({}),
            redundancy: dimension({}),
            leakage: dimension({ assessed: false, level: null }),
          },
        } as never,
      },
    })
    const by = (name: string) => bar.find(`[data-dimension="${name}"]`)
    expect(by('completeness').attributes('data-mark')).toBe('red')
    expect(by('completeness').find('[data-role="summary"]').text()).toBe('2 cols')
    expect(by('leakage').attributes('data-mark')).toBe('none')
    expect(by('leakage').find('[data-role="summary"]').text()).toBe('Not assessed')
  })

  it('says so when the run has no quality summary', () => {
    const bar = mount(HealthBar, { props: { tab: 'overview', dimensions: null } })
    expect(bar.find('[data-role="health-missing"]').exists()).toBe(true)
    expect(bar.findAll('li[data-mark="none"]')).toHaveLength(5)
  })
})
