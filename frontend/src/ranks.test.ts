import { describe, expect, it } from 'vitest'
import { confidenceRank, severityRank } from './generated/ranks'

describe('the rank tables', () => {
  it('order severities from info to error', () => {
    expect(severityRank('info')).toBeLessThan(severityRank('notice'))
    expect(severityRank('warn')).toBeLessThan(severityRank('error'))
  })

  it('order confidences from low to high', () => {
    expect(confidenceRank('low')).toBeLessThan(confidenceRank('moderate'))
    expect(confidenceRank('moderate')).toBeLessThan(confidenceRank('high'))
  })
})
