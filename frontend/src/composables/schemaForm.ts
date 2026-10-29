/** What a settings form needs from a JSON Schema: one field per property. */

export interface JsonSchema {
  $ref?: string
  type?: string
  enum?: unknown[]
  anyOf?: JsonSchema[]
  properties?: Record<string, JsonSchema>
  additionalProperties?: boolean | JsonSchema
  items?: JsonSchema
  default?: unknown
  description?: string
  title?: string
  minimum?: number
}

export type FieldKind =
  'string' | 'number' | 'integer' | 'boolean' | 'enum' | 'list' | 'group' | 'json'

export interface Field {
  name: string
  kind: FieldKind
  description: string
  default: unknown
  options: string[]
  fields: Field[]
}

export type Overlay = Record<string, unknown>

/** Settings the form leaves out: `dataset` is the key the settings belong to. */
const HIDDEN = new Set(['dataset'])

function resolve(schema: JsonSchema, root: Record<string, JsonSchema>): JsonSchema {
  let current = schema
  while (current.$ref) {
    const name = current.$ref.split('/').pop()!
    const target = root[name]
    if (!target) return current
    // Keep the property's own default and description over the referenced schema's.
    current = {
      ...target,
      default: schema.default ?? target.default,
      description: schema.description ?? target.description,
    }
  }
  // `X | None` is anyOf [X, null]: edit X, an empty field being null.
  if (current.anyOf) {
    const branches = current.anyOf.filter((branch) => branch.type !== 'null')
    if (branches.length === 1) {
      return resolve(
        { ...branches[0]!, description: current.description, default: current.default },
        root,
      )
    }
  }
  return current
}

function kindOf(schema: JsonSchema): FieldKind {
  if (schema.enum) return 'enum'
  if (schema.properties) return 'group'
  if (schema.type === 'boolean') return 'boolean'
  if (schema.type === 'integer') return 'integer'
  if (schema.type === 'number') return 'number'
  if (schema.type === 'string') return 'string'
  if (schema.type === 'array' && schema.items?.type === 'string') return 'list'
  return 'json'
}

/** The fields of an object schema; `root` holds the document's named schemas. */
export function fieldsOf(schema: JsonSchema, root: Record<string, JsonSchema>): Field[] {
  const properties = resolve(schema, root).properties ?? {}
  return Object.entries(properties)
    .filter(([name]) => !HIDDEN.has(name))
    .map(([name, property]) => {
      const resolved = resolve(property, root)
      const kind = kindOf(resolved)
      return {
        name,
        kind,
        description: resolved.description ?? '',
        default: resolved.default ?? null,
        options: kind === 'enum' ? resolved.enum!.map(String) : [],
        fields: kind === 'group' ? fieldsOf(resolved, root) : [],
      }
    })
}

/** A list field's text: comma-separated items. */
export function listText(value: unknown): string {
  return Array.isArray(value) ? value.join(', ') : ''
}

export function parseList(text: string): string[] {
  return text
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

/** The overlay without empty groups, so an untouched section is not saved. */
export function pruned(overlay: Overlay): Overlay {
  const out: Overlay = {}
  for (const [key, value] of Object.entries(overlay)) {
    if (value === undefined || value === null || value === '') continue
    if (typeof value === 'object' && !Array.isArray(value)) {
      const inner = pruned(value as Overlay)
      if (Object.keys(inner).length) out[key] = inner
      continue
    }
    out[key] = value
  }
  return out
}
