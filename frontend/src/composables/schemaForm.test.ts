import { describe, expect, it } from 'vitest'
import { fieldsOf, listText, parseList, pruned, type JsonSchema } from './schemaForm'

const root: Record<string, JsonSchema> = {
  Depth: { enum: ['quick', 'standard', 'deep'], type: 'string' },
  Limits: {
    type: 'object',
    properties: { rows: { type: 'integer', description: 'Most rows.' } },
  },
  Profile: {
    type: 'object',
    properties: {
      dataset: { type: 'string' },
      target: { anyOf: [{ type: 'string' }, { type: 'null' }], description: 'Predict this.' },
      depth: { $ref: '#/components/schemas/Depth', default: 'standard' },
      packs: { type: 'array', items: { type: 'string' }, default: ['core'] },
      limits: { $ref: '#/components/schemas/Limits', default: {} },
      params: { type: 'object', additionalProperties: true },
      workers: { anyOf: [{ type: 'integer', minimum: 1 }, { type: 'null' }] },
    },
  },
}

describe('fieldsOf', () => {
  const fields = fieldsOf({ $ref: '#/components/schemas/Profile' }, root)
  const by = (name: string) => fields.find((field) => field.name === name)!

  it('leaves out the dataset key', () => {
    expect(fields.map((field) => field.name)).not.toContain('dataset')
  })

  it('edits the non-null branch of an optional field', () => {
    expect(by('target')).toMatchObject({ kind: 'string', description: 'Predict this.' })
    expect(by('workers').kind).toBe('integer')
  })

  it('reads a referenced enum with the property default', () => {
    expect(by('depth')).toMatchObject({ kind: 'enum', default: 'standard' })
    expect(by('depth').options).toEqual(['quick', 'standard', 'deep'])
  })

  it('nests an object with properties and falls back to JSON for a free-form one', () => {
    expect(by('limits').kind).toBe('group')
    expect(by('limits').fields[0]).toMatchObject({ name: 'rows', kind: 'integer' })
    expect(by('params').kind).toBe('json')
    expect(by('packs').kind).toBe('list')
  })
})

describe('overlay text', () => {
  it('round-trips a list', () => {
    expect(parseList(' a, b ,,c ')).toEqual(['a', 'b', 'c'])
    expect(listText(['a', 'b'])).toBe('a, b')
    expect(listText(undefined)).toBe('')
  })

  it('prunes empty values and groups but keeps false and zero', () => {
    expect(
      pruned({ target: '', depth: 'quick', limits: { rows: undefined }, flag: false, n: 0 }),
    ).toEqual({ depth: 'quick', flag: false, n: 0 })
  })
})
