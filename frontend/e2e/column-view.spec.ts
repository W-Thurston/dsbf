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

test('the browser filters columns by name and type, in the URL', async ({ page, request }) => {
  const run = await runOf(request, 'titanic')
  const columns = (await (await request.get(`/v2/runs/${run}/columns`)).json()) as {
    name: string
    semantic_type: string
  }[]
  const numeric = columns.filter((c) => c.semantic_type === 'numeric')

  await page.goto(`/runs/${run}/columns`)
  await expect(page.getByRole('status')).toHaveText(`${columns.length} columns`)

  await page.getByLabel('Type').selectOption('numeric')
  await expect(page).toHaveURL(/type=numeric/)
  await expect(page.getByRole('status')).toHaveText(
    `${numeric.length} column${numeric.length === 1 ? '' : 's'}`,
  )

  await page.getByLabel('Name').fill('age')
  await page.reload()
  await expect(page.getByLabel('Name')).toHaveValue('age')
  await expect(page.getByRole('link', { name: 'age', exact: true })).toBeVisible()
})

test("age's histogram has the profile's bins and is drawn", async ({ page, request }) => {
  const run = await runOf(request, 'titanic')
  const age = (await (await request.get(`/v2/runs/${run}/profile?column=age`)).json()) as {
    histogram: { edges: number[]; counts: number[] }
  }

  await page.goto(`/runs/${run}/columns`)
  await page.getByRole('link', { name: 'age', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'age', level: 1 })).toBeVisible()
  await expect(page.getByText(`${age.histogram.counts.length} equal-width bins`)).toBeVisible()

  // The bin edges and counts reach the chart unchanged (see the chart option tests).
  await expect(
    page.locator('.dsbf-chart-card', { hasText: 'Histogram' }).locator('canvas'),
  ).toBeVisible()
})

test('a column that failed normality shows its Q-Q plot, and a passing one says why not', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  const result = (await (await request.get(`/v2/runs/${run}/checks/normality`)).json()) as {
    output: { columns: Record<string, { qq: unknown }> }
  }
  const flagged = Object.entries(result.output.columns).find(([, v]) => v.qq)?.[0]
  const passing = Object.entries(result.output.columns).find(([, v]) => !v.qq)?.[0]
  expect(flagged).toBeTruthy()

  await page.goto(`/runs/${run}/columns/${encodeURIComponent(flagged!)}`)
  await expect(page.getByText('Q-Q plot', { exact: true })).toBeVisible()
  await expect(
    page.locator('.dsbf-chart-card', { hasText: 'Q-Q plot' }).locator('canvas'),
  ).toBeVisible()

  if (passing) {
    await page.goto(`/runs/${run}/columns/${encodeURIComponent(passing)}`)
    await expect(page.getByText('so no Q-Q plot was kept')).toBeVisible()
  }
})

test('a run without a snapshot says so where the sample would be', async ({ page, request }) => {
  const bare = await runOf(request, 'bare')
  await page.goto(`/runs/${bare}/columns/age`)
  await expect(page.getByRole('heading', { name: 'Sample values' })).toBeVisible()
  await expect(page.getByText('kept no data snapshot')).toBeVisible()

  const full = await runOf(request, 'titanic')
  await page.goto(`/runs/${full}/columns/age`)
  await expect(page.getByRole('region', { name: 'Sample values' }).locator('tbody tr')).toHaveCount(
    20,
  )
})

test('a text column draws its top values and lists its findings', async ({ page, request }) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/columns/sex`)
  await expect(page.getByText('Top values', { exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Findings' })).toBeVisible()
})
