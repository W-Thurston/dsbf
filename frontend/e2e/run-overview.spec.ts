import { expect, test, type APIRequestContext } from '@playwright/test'

interface RunSummary {
  run_id: string
}

/** The fixture store's two runs: a full scan of titanic, and a sampled, failing one. */
async function fixtureRuns(request: APIRequestContext) {
  const { items } = (await (await request.get('/v2/runs')).json()) as { items: RunSummary[] }
  const overviews = await Promise.all(
    items.map(async ({ run_id }) => ({
      id: run_id,
      header: ((await (await request.get(`/v2/runs/${run_id}`)).json()) as { header: Header })
        .header,
    })),
  )
  const full = overviews.find((run) => run.header.dataset.sampling === null)
  const sampled = overviews.find((run) => run.header.dataset.sampling !== null)
  if (!full || !sampled) throw new Error('the fixture store lacks its two runs')
  return { full: full.id, sampled: sampled.id }
}

interface Header {
  dataset: { sampling: unknown }
}

test('the home page lists the dataset and links to its runs', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'Datasets' })).toBeVisible()
  await page
    .getByRole('region', { name: 'Datasets' })
    .getByRole('link', { name: 'titanic' })
    .click()
  await expect(page.getByRole('heading', { name: 'Runs of titanic' })).toBeVisible()
  await expect(page.getByRole('region', { name: 'Runs of titanic' }).getByRole('row')).toHaveCount(
    3,
  ) // header and two runs
})

test('opening titanic transfers under 200 KB', async ({ page, request }) => {
  const { full } = await fixtureRuns(request)
  let transferred = 0
  page.on('requestfinished', async (req) => {
    const sizes = await req.sizes()
    transferred += sizes.responseBodySize + sizes.responseHeadersSize
  })

  await page.goto(`/runs/${full}`)
  await expect(page.getByRole('heading', { name: 'Run health' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Top findings' })).toBeVisible()
  await page.waitForLoadState('networkidle')

  console.log(`opening titanic transferred ${(transferred / 1024).toFixed(1)} KB`)
  expect(transferred).toBeLessThan(200 * 1024)
})

test('a full scan has no sampling notice and no policy', async ({ page, request }) => {
  const { full } = await fixtureRuns(request)
  await page.goto(`/runs/${full}`)

  await expect(page.getByRole('heading', { name: 'Run health' })).toBeVisible()
  await expect(page.getByRole('note')).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'Policy' })).toHaveCount(0)
  await expect(page.getByText('not applicable').first()).toBeVisible()
})

test('a sampled, failing run shows its notice, verdict and every reason', async ({
  page,
  request,
}) => {
  const { sampled } = await fixtureRuns(request)
  await page.goto(`/runs/${sampled}`)

  await expect(page.getByRole('note')).toContainText('300 of 891 rows')
  const policy = page.getByRole('region', { name: 'Policy' })
  await expect(policy).toContainText('failed')
  await expect(policy).toContainText('no-broken-checks')

  const health = page.getByRole('region', { name: 'Run health' })
  await expect(health.locator('ul[data-status="error"]')).toContainText('always_fails')
  await expect(health.locator('ul[data-status="error"]')).toContainText('always fails')
  await expect(health.locator('ul[data-status="not_applicable"] li').first()).not.toBeEmpty()
  await expect(health.locator('ul[data-status="notes"]')).toContainText('plugin_load_failed')
})

test('the theme toggle switches the page between dark and light', async ({ page, request }) => {
  const { full } = await fixtureRuns(request)
  await page.goto(`/runs/${full}`)
  const body = page.locator('body')

  await expect(body).not.toHaveClass(/theme-light/)
  await page.getByRole('button', { name: /light theme/i }).click()
  await expect(body).toHaveClass(/theme-light/)
  await expect(page.getByRole('heading', { name: 'Run health' })).toBeVisible()
})

test('an unknown run says so', async ({ page }) => {
  await page.goto('/runs/00000000-0000-0000-0000-000000000000')

  await expect(page.locator('.error')).toContainText('00000000-0000-0000-0000-000000000000')
})
