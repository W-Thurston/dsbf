import { describe, expect, it } from 'vitest'
import {
  figureFor,
  figureForColumn,
  highlightHtml,
  qualityClass,
  qualityLabel,
  trunc,
} from './utils.js'

describe('highlightHtml', () => {
  it('wraps case-insensitive matches in <mark>, keeping the original casing', () => {
    expect(highlightHtml('Titanic run', 'TAN')).toBe('Ti<mark>tan</mark>ic run')
  })

  it('marks every occurrence', () => {
    expect(highlightHtml('a-b-a', 'a')).toBe('<mark>a</mark>-b-<mark>a</mark>')
  })

  it('escapes markup in the text', () => {
    expect(highlightHtml('<img src=x onerror=alert(1)>', '')).toBe(
      '&lt;img src=x onerror=alert(1)&gt;',
    )
  })

  it('escapes markup inside a match as well as around it', () => {
    expect(highlightHtml('a<b>c', '<b>')).toBe('a<mark>&lt;b&gt;</mark>c')
  })

  it('does not match inside an escaped entity', () => {
    expect(highlightHtml('a & b', 'amp')).toBe('a &amp; b')
  })

  it('treats regex metacharacters in the query literally', () => {
    expect(highlightHtml('price (usd) [1.5]', '(usd)')).toBe('price <mark>(usd)</mark> [1.5]')
    expect(highlightHtml('abc', '.')).toBe('abc')
    expect(highlightHtml('a.c', '.')).toBe('a<mark>.</mark>c')
  })

  it('returns the escaped text for a blank or missing query', () => {
    expect(highlightHtml('x < y', '   ')).toBe('x &lt; y')
    expect(highlightHtml('x', null)).toBe('x')
    expect(highlightHtml('x', undefined)).toBe('x')
  })

  it('renders null and undefined text as an empty string', () => {
    expect(highlightHtml(null, 'a')).toBe('')
    expect(highlightHtml(undefined, '')).toBe('')
  })

  it('stringifies non-string text', () => {
    expect(highlightHtml(2024, '02')).toBe('2<mark>02</mark>4')
  })
})

describe('qualityClass and qualityLabel', () => {
  it.each([
    [100, 'score-excellent', 'Excellent Quality'],
    [90, 'score-excellent', 'Excellent Quality'],
    [89.9, 'score-good', 'Good Quality'],
    [70, 'score-good', 'Good Quality'],
    [69.9, 'score-warn', 'Fair Quality'],
    [40, 'score-warn', 'Fair Quality'],
    [39.9, 'score-poor', 'Poor Quality'],
    [0, 'score-poor', 'Poor Quality'],
  ])('maps %s to %s and "%s"', (score, cls, label) => {
    expect(qualityClass(score)).toBe(cls)
    expect(qualityLabel(score)).toBe(label)
  })

  it('returns an empty string when there is no score', () => {
    expect(qualityClass(null)).toBe('')
    expect(qualityLabel(undefined)).toBe('')
  })
})

describe('trunc', () => {
  it('leaves short values alone and cuts long ones with an ellipsis', () => {
    expect(trunc('short')).toBe('short')
    expect(trunc('a'.repeat(20))).toBe('a'.repeat(20))
    expect(trunc('a'.repeat(21))).toBe('a'.repeat(20) + '…')
    expect(trunc('abcdef', 3)).toBe('abc…')
  })

  it('returns a dash for null and undefined, and stringifies numbers', () => {
    expect(trunc(null)).toBe('-')
    expect(trunc(undefined)).toBe('-')
    expect(trunc(12345, 3)).toBe('123…')
  })
})

describe('figure lookup', () => {
  const figures = [
    { column_name: null, plot_type: 'heatmap', format: 'static', theme: 'default', id: 1 },
    { column_name: null, plot_type: 'heatmap', format: 'static', theme: 'dark', id: 2 },
    { column_name: 'age', plot_type: 'hist', format: 'static', theme: 'dark', id: 3 },
    { column_name: 'age', plot_type: 'hist', format: 'static', theme: 'default', id: 4 },
    { column_name: 'fare', plot_type: 'hist', format: 'static', theme: 'default', id: 5 },
  ]

  it('prefers the requested theme', () => {
    expect(figureFor(figures, 'heatmap', 'static', 'dark').id).toBe(2)
    expect(figureForColumn(figures, 'age', 'hist', 'static', 'dark').id).toBe(3)
  })

  it('falls back to the theme-less figure', () => {
    expect(figureFor(figures, 'heatmap', 'static', 'light').id).toBe(1)
    expect(figureForColumn(figures, 'fare', 'hist', 'static', 'dark').id).toBe(5)
  })

  it('returns null when nothing matches', () => {
    expect(figureFor(figures, 'heatmap', 'interactive', 'dark')).toBeNull()
    expect(figureFor(figures, 'hist', 'static', 'dark')).toBeNull()
    expect(figureForColumn(figures, 'age', 'box', 'static', 'dark')).toBeNull()
  })
})
