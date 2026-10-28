import { describe, expect, it } from 'vitest'
import { PAGE_SIZE, apiQuery, parseFindingQuery, toLocationQuery } from './findingQuery'

describe('finding query', () => {
  it('reads filters from the URL and ignores values outside the vocabulary', () => {
    const query = parseFindingQuery({
      dimension: 'leakage',
      severity: 'warn',
      confidence: 'certain',
      sort: 'nonsense',
      offset: '-4',
      column: ['a', 'b'],
    })

    expect(query).toMatchObject({
      dimension: 'leakage',
      severity: 'warn',
      confidence: '',
      sort: 'severity',
      offset: 0,
      column: 'a',
    })
  })

  it('round-trips and leaves defaults out of the URL', () => {
    const query = parseFindingQuery({ severity: 'error', code: 'high_null_rate', offset: '50' })

    expect(toLocationQuery(query)).toEqual({
      severity: 'error',
      code: 'high_null_rate',
      offset: '50',
    })
    expect(toLocationQuery(parseFindingQuery({}))).toEqual({})
  })

  it('sends only the filters that are set to the endpoint', () => {
    const args = apiQuery(
      parseFindingQuery({ severity: 'warn', check: 'null_rates', offset: '25' }),
    )

    expect(args).toMatchObject({
      min_severity: 'warn',
      check: 'null_rates',
      dimension: undefined,
      sort: 'severity',
      limit: PAGE_SIZE,
      offset: 25,
    })
  })
})
