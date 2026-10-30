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

const detail = (page: import('@playwright/test').Page) =>
  page.locator('[data-section="column-detail"]')
const browser = (page: import('@playwright/test').Page) =>
  page.locator('[data-section="column-browser"]')

test('the browser groups columns by type, filters by name in the URL, and selects a column', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  const columns = (await (await request.get(`/v2/runs/${run}/columns`)).json()) as {
    name: string
    semantic_type: string
  }[]

  await page.goto(`/runs/${run}/distributions`)
  await expect(browser(page).getByRole('status')).toHaveText(`${columns.length} columns`)
  for (const type of new Set(columns.map((c) => c.semantic_type))) {
    await expect(browser(page).locator(`[data-group="${type}"]`)).toBeVisible()
  }
  // with no column in the path the first one is shown
  await expect(detail(page).getByRole('heading', { level: 1 })).toHaveText(columns[0]!.name)

  await page.getByLabel('Search columns').fill('age')
  await page.reload()
  await expect(page.getByLabel('Search columns')).toHaveValue('age')
  await browser(page).locator('[data-column="age"]').click()
  await expect(detail(page).getByRole('heading', { level: 1 })).toHaveText('age')
  await expect(page).toHaveURL(/columns\/age\?q=age/)
  // the search survives selecting a column
  await expect(page.getByLabel('Search columns')).toHaveValue('age')
})

test('a column with a warning finding has a dot, and every row has a null bar', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/distributions`)
  await expect(browser(page).locator('[data-role="null-bar"]').first()).toBeVisible()
  await expect(browser(page).locator('[data-role="warning-dot"]').first()).toBeVisible()
  // deck is mostly null in titanic
  await expect(
    browser(page).locator('[data-column="deck"] [data-role="null-bar"]'),
  ).toHaveAttribute('title', /^(7\d|8\d)/)
})

test('a numeric column shows its stats strip, histogram with a box plot toggle, and percentiles', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/columns/age`)
  const strip = detail(page).locator('[data-section="stats"]')
  for (const label of [
    'Count',
    'Mean',
    'Median',
    'Std',
    'Skew',
    'Kurtosis',
    'IQR',
    'Null',
    'Outliers',
  ]) {
    await expect(strip.locator(`[data-stat="${label}"]`)).toBeVisible()
  }
  const chart = detail(page).locator('[data-chart="distribution"]')
  await expect(chart.locator('canvas')).toBeVisible()
  await expect(chart.getByText(/equal-width bins/)).toBeVisible()

  await chart.getByRole('button', { name: 'Box plot' }).click()
  await expect(chart.getByText('Box plot', { exact: true }).first()).toBeVisible()
  await chart.getByRole('button', { name: 'Histogram' }).click()
  await expect(chart.getByText(/equal-width bins/)).toBeVisible()

  await expect(detail(page).getByText('Percentiles')).toBeVisible()
})

test('outliers list each method with a consensus line, and normality its tests', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/columns/fare`)
  const outliers = detail(page).locator('[data-section="outliers"]')
  await expect(outliers.locator('[data-role="consensus"]')).toBeVisible()
  await expect(outliers.getByRole('row', { name: /IQR/ })).toBeVisible()
  await expect(outliers.getByRole('row', { name: /Z-score/ })).toBeVisible()
  await expect(outliers.getByRole('row', { name: /MAD/ })).toBeVisible()
  await expect(detail(page).locator('[data-section="normality"]')).toContainText('Not normal')
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
  await expect(detail(page).locator('[data-chart="qq"] canvas')).toBeVisible()

  if (passing) {
    await page.goto(`/runs/${run}/columns/${encodeURIComponent(passing)}`)
    await expect(page.getByText('so no Q-Q plot was kept')).toBeVisible()
  }
})

test('a categorical column draws its top values, value counts and findings', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/columns/sex`)
  await expect(detail(page).locator('[data-chart="top-values"] canvas')).toBeVisible()
  await expect(detail(page).locator('[data-stat="Top value"]')).toContainText('male')
  await expect(detail(page).locator('[data-section="values-table"]')).toContainText('female')
  await expect(detail(page).locator('[data-section="guidance"]')).toBeVisible()
  await expect(detail(page).locator('[data-section="normality"]')).toHaveCount(0)
})

test('a bimodal column has a callout and a toggle for the fitted components', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'kitchen5k')
  await page.goto(`/runs/${run}/columns/bimodal_column`)
  await expect(detail(page).locator('[data-role="bimodal-callout"]')).toContainText('Two peaks')
  const toggle = detail(page).getByRole('button', { name: 'Show fitted components' })
  await expect(toggle).toHaveAttribute('aria-pressed', 'false')
  await toggle.click()
  await expect(toggle).toHaveAttribute('aria-pressed', 'true')
  await expect(detail(page).locator('[data-chart="distribution"] canvas')).toBeVisible()

  // a unimodal column has neither
  await page.goto(`/runs/${run}/columns/normal_numeric_int`)
  await expect(detail(page).locator('[data-role="bimodal-callout"]')).toHaveCount(0)
  await expect(detail(page).getByRole('button', { name: 'Show fitted components' })).toHaveCount(0)
})

test('a top correlation selects that column', async ({ page, request }) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/columns/fare`)
  const first = detail(page).locator('[data-section="correlations"] a').first()
  const other = (await first.textContent())!.trim()
  await first.click()
  await expect(detail(page).getByRole('heading', { level: 1 })).toHaveText(other)
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

// With SCREENSHOT_DIR set, save the tab on a numeric, categorical and bimodal column.
test('save the distributions screenshots', async ({ page, request }) => {
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1400, height: 1500 })
  const shots: [string, string, string][] = [
    ['titanic', 'age', 'age'],
    ['titanic', 'sex', 'sex'],
    ['kitchen5k', 'bimodal_column', 'bimodal'],
  ]
  for (const [dataset, column, name] of shots) {
    const run = await runOf(request, dataset)
    for (const theme of ['dark', 'light']) {
      await page.goto(`/runs/${run}/columns/${column}`)
      const toggle = page.getByRole('button', { name: /theme/ })
      if (theme === 'light') await toggle.click()
      if (name === 'bimodal')
        await page.getByRole('button', { name: 'Show fitted components' }).click()
      await expect(detail(page).locator('canvas').first()).toBeVisible()
      await page.waitForTimeout(2000)
      await page.screenshot({ path: `${dir}/distributions-${name}-${theme}.png`, fullPage: true })
    }
  }
})
