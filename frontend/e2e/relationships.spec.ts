import { expect, test, type APIRequestContext } from '@playwright/test'

interface Summary {
  run_id: string
  dataset_key: string
}

async function runOf(request: APIRequestContext, dataset: string) {
  const { items } = (await (await request.get('/v2/runs')).json()) as { items: Summary[] }
  const found = items.find((run) => run.dataset_key === dataset)
  if (!found) throw new Error(`the fixture store has no ${dataset} run`)
  return found.run_id
}

test("the matrix panels and table list exactly the checks' tested pairs", async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  const matrix = (await (await request.get(`/v2/runs/${run}/relationships`)).json()) as {
    cells: { statistic: string }[]
    coverage: { pairs_tested: number }[]
  }
  expect(matrix.cells.length).toBe(matrix.coverage.reduce((n, c) => n + c.pairs_tested, 0))

  await page.goto(`/runs/${run}/relationships`)
  await expect(page.getByRole('heading', { name: 'Relationships', level: 1 })).toBeVisible()
  await expect(page.getByLabel('Coverage').getByText('associations')).toBeVisible()
  await expect(page.getByRole('region', { name: 'Tested pairs' }).locator('tbody tr')).toHaveCount(
    matrix.cells.length,
  )
  await expect(
    page.locator('.dsbf-chart-card', { hasText: 'Correlation' }).locator('canvas').first(),
  ).toBeVisible()
})

test('the table groups pairs by statistic before value', async ({ page, request }) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/relationships`)
  const statistics = await page
    .getByRole('region', { name: 'Tested pairs' })
    .locator('tbody tr td:first-child')
    .allTextContents()
  // Each statistic appears in one unbroken run of rows.
  const runs = statistics.filter((s, i) => i === 0 || s !== statistics[i - 1])
  expect(new Set(runs).size).toBe(runs.length)
})

test('choosing a pair draws it from the snapshot and says how it was sampled', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/relationships?first=age&second=fare`)
  await expect(page.getByRole('status').filter({ hasText: 'from a snapshot of' })).toBeVisible()
  await expect(
    page
      .locator('.dsbf-chart-card')
      .filter({ has: page.locator('canvas') })
      .last(),
  ).toBeVisible()
})

test('a run without a snapshot says so under the pair plot', async ({ page, request }) => {
  const bare = await runOf(request, 'bare')
  await page.goto(`/runs/${bare}/relationships?first=age&second=fare`)
  await expect(page.getByRole('alert')).toContainText('kept no data snapshot')
})
