// Fail when the built dashboard outgrows its budget (run after `pnpm build`).
//
// Set from the first build of the rebuilt shell (2026-10-28): the entry chunk was
// 96.7 KB and the whole bundle 120 KB. A tab unit that needs more raises the budget
// here, in the same commit, so growth is a decision. The column view (2026-10-28) is
// the first to draw a chart, so the total now includes the lazy chunk that holds
// ECharts (about 700 KB, 236 KB gzipped, fetched only on a page with a chart); the
// entry budget, which every page pays, is unchanged.
import { readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

const KB = 1024
const BUDGET = { entry: 110 * KB, total: 880 * KB }
const assets = join(import.meta.dirname, '..', 'dist', 'assets')

const files = readdirSync(assets).map((name) => ({ name, size: statSync(join(assets, name)).size }))
const entry = files.find((f) => /^index-.*\.js$/.test(f.name))
const total = files.reduce((sum, f) => sum + f.size, 0)

const problems = []
if (!entry) problems.push('no entry chunk in dist/assets: run `pnpm build` first')
else if (entry.size > BUDGET.entry)
  problems.push(`entry chunk ${(entry.size / KB).toFixed(1)} KB > ${BUDGET.entry / KB} KB`)
if (total > BUDGET.total)
  problems.push(`bundle ${(total / KB).toFixed(1)} KB > ${BUDGET.total / KB} KB`)

console.log(
  `entry ${((entry?.size ?? 0) / KB).toFixed(1)} KB, bundle ${(total / KB).toFixed(1)} KB`,
)
if (problems.length) {
  console.error(problems.join('\n'))
  process.exit(1)
}
