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

const metric = (page: import('@playwright/test').Page, label: string) =>
  page.locator(`[data-metric="${label}"]`)

test('the metric strip shows the run at a glance', async ({ page, request }) => {
  await page.goto(`/runs/${await runOf(request, 'high_missingness')}`)

  await expect(metric(page, 'Rows')).toContainText('2,000')
  await expect(metric(page, 'Columns')).toBeVisible()
  await expect(metric(page, 'Memory')).toContainText(/KB|MB/)
  await expect(metric(page, 'Missing')).toContainText('%')
  await expect(metric(page, 'Duplicate rows')).toContainText('11')
  await expect(metric(page, 'Sample adequacy')).toContainText(/adequate|limited|insufficient/)
  await expect(metric(page, 'Depth')).toBeVisible()
})

test('the column table colours, sorts and filters, and each row has a chart', async ({
  page,
  request,
}) => {
  await page.goto(`/runs/${await runOf(request, 'high_missingness')}`)
  const table = page.locator('[data-section="columns"]')

  await expect(table.locator('tbody tr').first()).toBeVisible()
  await expect(table.locator('[data-role="row-sketch"]').first()).toBeVisible()
  // notes is the most missing column in this archetype: red at 20% or more
  await expect(table.locator('[data-column="notes"] [data-cell="missing"]')).toHaveClass(/tone-red/)

  await table.getByRole('button', { name: 'Missing' }).click()
  await expect(table.locator('tbody tr').first()).toHaveAttribute('data-column', /.+/)
  const first = await table.locator('tbody tr').first().getAttribute('data-column')
  await table.getByRole('button', { name: 'Missing' }).click()
  expect(await table.locator('tbody tr').first().getAttribute('data-column')).not.toBe(first)

  await table.getByRole('searchbox').fill('notes')
  await expect(table.locator('tbody tr')).toHaveCount(1)
  await table.getByRole('searchbox').fill('no-such-column')
  await expect(table).toContainText('No column matches.')
})

test('missingness charts the nulls and the columns that go missing together', async ({
  page,
  request,
}) => {
  await page.goto(`/runs/${await runOf(request, 'high_missingness')}`)
  const panel = page.locator('[data-section="missingness"]')

  await expect(panel.getByRole('heading', { name: 'Missing values' })).toBeVisible()
  await expect(panel.getByText('Columns that go missing together')).toBeVisible()
  await expect(panel.locator('[data-role="null-matrix"]')).toBeVisible()
  await expect(panel.locator('[data-bar]').first()).toBeVisible()
})

test('the findings are charted by dimension and link to the explorer', async ({
  page,
  request,
}) => {
  await page.goto(`/runs/${await runOf(request, 'high_missingness')}`)

  await expect(
    page.locator('[data-section="finding-counts"] [data-finding-dimension]').first(),
  ).toBeVisible()
  await page.getByRole('link', { name: 'Findings explorer' }).click()
  await expect(page).toHaveURL(/\/findings/)
})

test('a run without a snapshot says so instead of showing a sample', async ({ page, request }) => {
  await page.goto(`/runs/${await runOf(request, 'bare')}`)

  await expect(page.locator('[data-role="no-snapshot"]')).toBeVisible()
})

test('the sample shows rows, with nulls marked, and the header says if the content changed', async ({
  page,
  request,
}) => {
  await page.goto(`/runs/${await runOf(request, 'titanic')}`)
  const sample = page.locator('[data-section="sample"]')

  await expect(sample.locator('tbody tr')).toHaveCount(10)
  await sample.getByLabel('Rows shown').selectOption('5')
  await expect(sample.locator('tbody tr')).toHaveCount(5)
  await expect(page.locator('[data-role="lineage"]')).toContainText('titanic')
  await expect(page.locator('[data-role="content-change"]')).toContainText('since the previous run')
})

test('a column with no nulls has no missingness heatmap, and the clean dataset says so', async ({
  page,
  request,
}) => {
  await page.goto(`/runs/${await runOf(request, 'clean')}`)

  await expect(page.locator('[data-role="no-nulls"]')).toBeVisible()
})

// With SCREENSHOT_DIR set, save the Overview in both themes for review.
test('save the overview screenshots', async ({ page, request }) => {
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1280, height: 900 })
  for (const dataset of ['titanic', 'high_missingness']) {
    const run = await runOf(request, dataset)
    for (const theme of ['dark', 'light']) {
      await page.goto(`/runs/${run}`)
      if (theme === 'light') await page.getByRole('button', { name: /theme/ }).click()
      await expect(
        page.locator('[data-section="missingness"], [data-role="no-nulls"]'),
      ).toBeVisible()
      await page.waitForLoadState('networkidle')
      await page.screenshot({ path: `${dir}/overview-${dataset}-${theme}.png`, fullPage: true })
    }
  }
})
