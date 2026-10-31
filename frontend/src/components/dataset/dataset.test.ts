// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { diffCounts } from '../compare/diffCounts'
import DimensionDots from './DimensionDots.vue'
import { dimensionMarks } from './dimensionMarks'

describe('dimension dots', () => {
  it('gives every dimension a mark, with the empty ring for one not reported or not assessed', () => {
    const marks = dimensionMarks({ completeness: 'amber', validity: null })
    expect(marks.map((m) => m.name)).toEqual([
      'completeness',
      'validity',
      'usability',
      'redundancy',
      'leakage',
    ])
    expect(marks.map((m) => m.mark)).toEqual(['amber', 'none', 'none', 'none', 'none'])
  })

  it('names each dimension and its state for a screen reader', () => {
    const wrapper = mount(DimensionDots, { props: { levels: { leakage: 'red' } } })
    const leakage = wrapper.get('[data-dimension="leakage"]')
    expect(leakage.attributes('data-mark')).toBe('red')
    expect(leakage.text()).toContain('Leakage: serious')
    expect(wrapper.get('[data-dimension="completeness"]').text()).toContain('not assessed')
  })
})

describe('diff counts', () => {
  it('counts each kind of change', () => {
    const counts = diffCounts({
      new: [1, 2],
      resolved: [],
      changed: [1],
      status_changes: [1, 2, 3],
    } as never)
    expect(counts.map((c) => c.count)).toEqual([2, 0, 1, 3])
  })
})
