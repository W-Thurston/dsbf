/**
 * Generated from dsbf.model.vocabulary by `dsbf schema export`.
 * Do not edit by hand: change the enums and regenerate.
 */

export const SEVERITY_ORDER = ["info", "notice", "warn", "error"] as const

export type SeverityName = (typeof SEVERITY_ORDER)[number]

/** Position in `SEVERITY_ORDER`, lowest first. */
export function severityRank(value: SeverityName): number {
  return SEVERITY_ORDER.indexOf(value)
}

export const CONFIDENCE_ORDER = ["low", "moderate", "high"] as const

export type ConfidenceName = (typeof CONFIDENCE_ORDER)[number]

/** Position in `CONFIDENCE_ORDER`, lowest first. */
export function confidenceRank(value: ConfidenceName): number {
  return CONFIDENCE_ORDER.indexOf(value)
}
