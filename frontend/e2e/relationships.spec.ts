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

test('the summary charts the strength bands and a collinearity chip narrows the pairs', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'severe_multicollinearity')
  await page.goto(`/runs/${run}/relationships`)
  await expect(page.locator('[data-role="strength-bar"]')).toBeVisible()
  await expect(page.getByLabel('Warnings').getByRole('button')).not.toHaveCount(0)
  const all = await page.getByRole('region', { name: 'Tested pairs' }).locator('tbody tr').count()
  await page.locator('[data-chip="collinearity"]').click()
  await expect(page).toHaveURL(/chip=collinearity/)
  const narrowed = await page
    .getByRole('region', { name: 'Tested pairs' })
    .locator('tbody tr')
    .count()
  expect(narrowed).toBeLessThanOrEqual(all)
  await expect(page.locator('[data-role="multivariate-outliers"]')).toBeVisible()
})

test('picking a column lists its associations and the strength filter keeps the URL', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/relationships`)
  await page.locator('[data-column="age"]').click()
  await expect(page).toHaveURL(/column=age/)
  await expect(page.getByRole('heading', { name: 'Associations of age' })).toBeVisible()
  const rows = page.getByRole('region', { name: /Associations of age/ }).locator('tbody tr')
  expect(await rows.count()).toBeGreaterThan(0)
  await page.getByLabel('Strength').selectOption('strong')
  await expect(page).toHaveURL(/strength=strong/)
})

test('a wide dataset caps each heatmap panel and says so', async ({ page, request }) => {
  const run = await runOf(request, 'wide')
  await page.goto(`/runs/${run}/relationships`)
  await expect(page.getByText(/30 of \d+ columns, strongest first/).first()).toBeVisible()
})

test('an all-categorical dataset draws only the category panel', async ({ page, request }) => {
  const run = await runOf(request, 'all_categorical')
  await page.goto(`/runs/${run}/relationships`)
  await expect(page.locator('.dsbf-chart-card', { hasText: 'Category by category' })).toBeVisible()
  await expect(page.locator('.dsbf-chart-card', { hasText: 'Correlation' })).toHaveCount(0)
})

test('a single-column dataset says no pair was tested', async ({ page, request }) => {
  const run = await runOf(request, 'single_column')
  await page.goto(`/runs/${run}/relationships`)
  await expect(page.getByRole('status').filter({ hasText: 'No pair was tested' })).toBeVisible()
})

// With SCREENSHOT_DIR set, save the tab in both themes for review.
test('save the relationships screenshots', async ({ page, request }) => {
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1280, height: 900 })
  for (const dataset of ['titanic', 'severe_multicollinearity']) {
    const run = await runOf(request, dataset)
    for (const theme of ['dark', 'light']) {
      await page.goto(`/runs/${run}/relationships`)
      if (theme === 'light') await page.getByRole('button', { name: /theme/ }).click()
      await expect(page.locator('[data-role="strength-bar"]')).toBeVisible()
      await page.waitForTimeout(800)
      await page.screenshot({
        path: `${dir}/relationships-${dataset}-${theme}.png`,
        fullPage: true,
      })
    }
  }
})
