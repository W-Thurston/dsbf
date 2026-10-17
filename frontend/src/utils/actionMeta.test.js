import { describe, expect, it } from 'vitest'
import {
  ACTION_META,
  SKIP_CONTEXT,
  actionMeta,
  enrichSkipReason,
  normalizeMethod,
} from './actionMeta.js'

describe('normalizeMethod', () => {
  it('lower-cases and turns hyphens, dots and spaces into underscores', () => {
    expect(normalizeMethod('Box-Cox')).toBe('box_cox')
    expect(normalizeMethod('Yeo.Johnson')).toBe('yeo_johnson')
    expect(normalizeMethod('log1p  transform')).toBe('log1p_transform')
  })

  it('treats null and undefined as an empty string', () => {
    expect(normalizeMethod(null)).toBe('')
    expect(normalizeMethod(undefined)).toBe('')
  })
})

describe('enrichSkipReason', () => {
  it('expands a known reason', () => {
    expect(enrichSkipReason('requires all values > 0')).toBe(
      SKIP_CONTEXT['requires all values > 0'],
    )
  })

  it('passes an unknown reason through', () => {
    expect(enrichSkipReason('column is constant')).toBe('column is constant')
  })

  it('supplies a generic message when there is no reason', () => {
    expect(enrichSkipReason(undefined)).toBe('This transform could not be applied to this column.')
  })
})

describe('actionMeta', () => {
  const [key, meta] = Object.entries(ACTION_META)[0]

  it('looks an action up by method, normalising the key', () => {
    expect(actionMeta({ method: key.toUpperCase().replace(/_/g, '-') })).toBe(meta)
  })

  it('falls back to the action name when the method is unknown', () => {
    expect(actionMeta({ method: 'no-such-method', action: key })).toBe(meta)
  })

  it('returns null when nothing matches', () => {
    expect(actionMeta({ method: 'no-such-method', action: 'no-such-action' })).toBeNull()
    expect(actionMeta({})).toBeNull()
  })
})
