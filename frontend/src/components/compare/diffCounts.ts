import type { components } from '../../generated/api'

/** Counts per kind of change, in the order the sections below the chart list them. */
export function diffCounts(diff: components['schemas']['RunDiff']) {
  return [
    { label: 'New findings', count: diff.new.length },
    { label: 'Resolved findings', count: diff.resolved.length },
    { label: 'Changed findings', count: diff.changed.length },
    { label: 'Status changes', count: diff.status_changes.length },
  ]
}
