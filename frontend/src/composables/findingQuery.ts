import type { components } from '../generated/api'
import type { LocationQuery, LocationQueryRaw } from 'vue-router'
import {
  CONFIDENCE_ORDER,
  SEVERITY_ORDER,
  type ConfidenceName,
  type SeverityName,
} from '../generated/ranks'

export const SORTS = ['severity', 'check', 'code'] as const
export type FindingSortName = (typeof SORTS)[number]

export const PAGE_SIZE = 25

/** What the explorer shows: every part of it lives in the URL's query string. */
export interface FindingQuery {
  dimension: string
  severity: SeverityName | ''
  confidence: ConfidenceName | ''
  check: string
  code: string
  column: string
  sort: FindingSortName
  offset: number
  finding: string
}

function text(value: LocationQuery[string] | undefined): string {
  return (Array.isArray(value) ? value[0] : value) ?? ''
}

function oneOf<T extends string>(allowed: readonly T[], value: string): T | '' {
  return allowed.find((candidate) => candidate === value) ?? ''
}

export function parseFindingQuery(query: LocationQuery): FindingQuery {
  const offset = Number.parseInt(text(query.offset), 10)
  return {
    dimension: text(query.dimension),
    severity: oneOf(SEVERITY_ORDER, text(query.severity)),
    confidence: oneOf(CONFIDENCE_ORDER, text(query.confidence)),
    check: text(query.check),
    code: text(query.code),
    column: text(query.column),
    sort: oneOf(SORTS, text(query.sort)) || 'severity',
    offset: Number.isFinite(offset) && offset > 0 ? offset : 0,
    finding: text(query.finding),
  }
}

/** The query string for `query`, leaving out whatever is at its default. */
export function toLocationQuery(query: FindingQuery): LocationQueryRaw {
  const raw: LocationQueryRaw = {}
  for (const key of ['dimension', 'severity', 'confidence', 'check', 'code', 'column'] as const) {
    if (query[key]) raw[key] = query[key]
  }
  if (query.sort !== 'severity') raw.sort = query.sort
  if (query.offset > 0) raw.offset = String(query.offset)
  if (query.finding) raw.finding = query.finding
  return raw
}

/** The arguments of the findings endpoint for `query`. */
export function apiQuery(query: FindingQuery) {
  return {
    dimension: (query.dimension || undefined) as components['schemas']['Dimension'] | undefined,
    min_severity: query.severity || undefined,
    min_confidence: query.confidence || undefined,
    check: query.check || undefined,
    code: query.code || undefined,
    column: query.column || undefined,
    sort: query.sort,
    limit: PAGE_SIZE,
    offset: query.offset,
  } as const
}
