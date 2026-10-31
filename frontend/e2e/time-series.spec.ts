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

  await page.locator('[data-section="column-browser"] [data-column]').nth(1).click()
  await expect(page.locator('[data-section="decomposition"] canvas')).toHaveCount(1)
  expect(pages[1]).toContain('offset=365')
})

test('the header strip names the index, the frequency and the value columns', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'kitchen5k')
  await page.goto(`/runs/${run}/time-series`)
  const strip = page.locator('[data-role="series"]')
  await expect(strip.locator('[data-role="index"]')).not.toBeEmpty()
  await expect(strip.locator('[data-role="frequency"]')).toContainText(/day|daily|D/i)
  await expect(strip.locator('[data-role="periods"]')).toContainText('365')
  await expect(strip.locator('[data-role="value-columns"]')).toHaveText(/^\d+$/)
})

test('the selected column is in the URL and survives a reload', async ({ page, request }) => {
  const run = await runOf(request, 'kitchen5k')
  await page.goto(`/runs/${run}/time-series`)
  const second = page.locator('[data-section="column-browser"] [data-column]').nth(1)
  const name = await second.getAttribute('data-column')
  await second.click()
  await expect(page).toHaveURL(new RegExp(`column=${name}`))
  await page.reload()
  await expect(
    page.locator(`[data-section="column-browser"] [data-column="${name}"]`),
  ).toHaveAttribute('aria-current', 'true')
  await expect(page.locator(`[data-section="stationarity"] [data-column="${name}"]`)).toHaveClass(
    /selected/,
  )
})

test('the decomposition shows trend and seasonal strength against the noise floor', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'kitchen5k')
  await page.goto(`/runs/${run}/time-series`)
  const strengths = page.locator('[data-role="strengths"]')
  await expect(strengths.locator('[data-name="Trend"]')).toBeVisible()
  await expect(strengths.locator('[data-name="Seasonal"]')).toBeVisible()
  await expect(page.locator('[data-role="lags"]')).toContainText('Significant ACF lags')
})

// With SCREENSHOT_DIR set, save the tab in both themes for review.
test('save the time series screenshots', async ({ page, request }) => {
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1280, height: 900 })
  for (const dataset of ['kitchen5k', 'titanic']) {
    const run = await runOf(request, dataset)
    for (const theme of ['dark', 'light']) {
      await page.goto(`/runs/${run}/time-series`)
      if (theme === 'light') await page.getByRole('button', { name: /theme/ }).click()
      await expect(
        page.locator(
          dataset === 'titanic'
            ? '[data-role="empty-state"]'
            : '[data-section="decomposition"] canvas',
        ),
      ).toBeVisible()
      await page.screenshot({ path: `${dir}/time-series-${dataset}-${theme}.png`, fullPage: true })
    }
  }
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
