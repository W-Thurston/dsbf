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

test('a run with a time index shows stationarity, the correlograms and the four STL series', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'kitchen5k')
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.goto(`/runs/${run}/time-series`)
  await expect(page.getByRole('heading', { name: 'Time series', level: 1 })).toBeVisible()
  await expect(page.locator('[data-role="empty-state"]')).toHaveCount(0)

  const stationarity = page.locator('[data-section="stationarity"]')
  await expect(stationarity.locator('tbody tr').first()).toBeVisible()
  await expect(stationarity.locator('[data-column="normal_numeric_int"]')).toContainText(
    'Stationary',
  )

  const acf = page.locator('[data-section="autocorrelation"]')
  await expect(acf).toContainText('Ljung-Box p')
  await expect(acf.locator('canvas')).toHaveCount(2)

  const stl = page.locator('[data-section="decomposition"]')
  await expect(stl).toContainText('Cycle of 7 periods')
  await expect(stl.locator('canvas')).toHaveCount(1)
  expect(errors).toEqual([])
})

test('the STL series are the stored artifact rows of the selected column', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'kitchen5k')
  const pages: string[] = []
  page.on('request', (r) => {
    if (r.url().includes('/artifacts/seasonal_decomposition')) pages.push(r.url())
  })
  await page.goto(`/runs/${run}/time-series`)
  await expect(page.locator('[data-section="decomposition"] canvas')).toHaveCount(1)
  // 365 daily periods of the first column: one request, from row 0.
  expect(pages).toHaveLength(1)
  expect(pages[0]).toContain('offset=0')
  expect(pages[0]).toContain('limit=365')

  await page.locator('[data-role="column"]').selectOption({ index: 1 })
  await expect(page.locator('[data-section="decomposition"] canvas')).toHaveCount(1)
  expect(pages[1]).toContain('offset=365')
})

test('a run with no time index says how to set one', async ({ page, request }) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/time-series`)
  const empty = page.locator('[data-role="empty-state"]')
  await expect(empty).toBeVisible()
  await expect(empty).toContainText('time_index')
  await expect(empty).toContainText('--time-index')
  await expect(page.locator('[data-section="stationarity"]')).toHaveCount(0)
})

test('the run overview links to the tab', async ({ page, request }) => {
  const run = await runOf(request, 'kitchen5k')
  await page.goto(`/runs/${run}`)
  await page.getByRole('link', { name: 'Time series' }).click()
  await expect(page).toHaveURL(new RegExp(`/runs/${run}/time-series$`))
})
