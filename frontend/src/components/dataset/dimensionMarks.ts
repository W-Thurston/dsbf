import { DIMENSIONS, type Mark } from '../shell/health'

/** One mark per dimension in the health bar's order; a dimension the run did not
 * report, or did not assess, is the empty ring. */
export function dimensionMarks(levels: Record<string, Mark | null>) {
  return DIMENSIONS.map(({ name, label }) => ({
    name,
    label,
    mark: (levels[name] ?? 'none') as Mark,
  }))
}
