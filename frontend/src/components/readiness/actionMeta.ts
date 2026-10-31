// What a suggested action means, as plain-language notes the dashboard shows beside the
// finding's own detail. The dictionary is data (`actionMeta.json`) so a Python test can
// check it against the guidance the engine can emit; a pair with no entry is listed there.

import dictionary from './actionMeta.json'

export interface ActionNote {
  what: string
  tradeoff: string
  after: string
  ref: { label: string; url: string } | null
}

const entries = dictionary.entries as Record<string, ActionNote>

const normal = (text: string) => text.toLowerCase().replace(/[-.\s]+/g, '_')

/**
 * The note for an action: `action:method` first, then the method alone. An action that
 * carries a method is never matched on the action alone, as `stratified_split` with a
 * `time_ordered` method is not a stratified split.
 */
export function actionNote(action: string, method?: string | null): ActionNote | null {
  if (!method) return entries[normal(action)] ?? null
  const key = normal(method)
  return entries[`${normal(action)}:${key}`] ?? entries[key] ?? null
}
