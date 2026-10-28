/** Display helpers shared by the views. */

const COUNT = new Intl.NumberFormat('en-US')

export function formatCount(value: number): string {
  return COUNT.format(value)
}

/** An ISO timestamp as the viewer's local date and time, or the input when it is not one. */
export function formatDateTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function plural(count: number, noun: string): string {
  return `${formatCount(count)} ${noun}${count === 1 ? '' : 's'}`
}

/** "not_applicable" as "not applicable". */
export function words(value: string): string {
  return value.replaceAll('_', ' ')
}
