import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

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

const card = (page: Page, id: string) => page.locator(`[data-dimension="${id}"]`)
const level = (page: Page, id: string) => card(page, id).locator('[data-role="level"]')

test('a clean dataset is ready, and a dimension without a target is not assessed, not green', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'clean')
  await page.goto(`/runs/${run}/quality`)
  await expect(page.locator('[data-role="trust"]')).toContainText('Looking good')
  for (const id of [
    'quality-completeness',
    'quality-validity',
    'quality-usability',
    'quality-redundancy',
  ]) {
    await expect(level(page, id)).toHaveText('All clear')
  }
  await expect(level(page, 'quality-leakage')).toHaveText('Not assessed')
  await expect(page.locator('[data-section="leakage"] [data-role="not-assessed"]')).toBeVisible()
  await expect(level(page, 'quality-leakage')).not.toHaveClass(/badge-ok/)

  await page.goto(`/runs/${run}/ml-readiness`)
  await expect(page.locator('[data-role="gate"]')).toContainText('Ready for modelling')
  await expect(page.locator('[data-role="gate"]')).toContainText('Leakage risk')
  for (const id of ['ml-transformations', 'ml-encoding', 'ml-missingness', 'ml-unusable']) {
    await expect(level(page, id)).toHaveText('All clear')
  }
  await expect(level(page, 'ml-leakage')).toHaveText('Not assessed')
  await expect(page.locator('[data-section="leakage"] [data-role="not-assessed"]')).toBeVisible()
  await expect(level(page, 'ml-leakage')).not.toHaveClass(/badge-ok/)
})

test('every column of a clean dataset is listed as clean', async ({ page, request }) => {
  const run = await runOf(request, 'clean')
  const columns = (await (await request.get(`/v2/runs/${run}/columns`)).json()) as unknown[]
  await page.goto(`/runs/${run}/quality`)
  await expect(page.locator('[data-role="clean-columns"] li')).toHaveCount(columns.length)
})

test('missing values make completeness serious and the dataset not ready', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/ml-readiness`)
  await expect(page.locator('[data-role="gate"]')).toContainText('Not ready for modelling')
  await expect(level(page, 'ml-missingness')).toHaveText('Serious')
  await page.goto(`/runs/${run}/quality`)
  await expect(level(page, 'quality-completeness')).toHaveText('Serious')
  const completeness = page.locator('[data-section="completeness"]')
  for (const column of ['device_type', 'premium_score', 'notes']) {
    await expect(completeness).toContainText(column)
  }
})

test('a readiness finding offers its actions and links to the findings explorer', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/ml-readiness`)
  const missingness = page.locator('[data-section="missingness"]')
  await missingness.locator('tr.row').first().getByRole('button').click()
  await expect(missingness.locator('.guidance li').first()).toBeVisible()
  await missingness.getByRole('link', { name: 'Open in the findings explorer' }).first().click()
  await expect(page).toHaveURL(new RegExp(`/runs/${run}/findings\\?`))
})

test('coverage by column matches the API and every column is listed', async ({ page, request }) => {
  const run = await runOf(request, 'high_missingness')
  const coverage = (await (await request.get(`/v2/runs/${run}/coverage`)).json()) as {
    columns: { column: string; examined_by: number; checks_completed: number }[]
  }
  await page.goto(`/runs/${run}/quality`)
  const rows = page.getByRole('region', { name: 'Coverage by column' }).locator('tbody tr')
  await expect(rows).toHaveCount(coverage.columns.length)
  const first = coverage.columns[0]!
  await expect(rows.first()).toContainText(`${first.examined_by} of ${first.checks_completed}`)
})

test('the run overview links to the tab', async ({ page, request }) => {
  const run = await runOf(request, 'clean')
  await page.goto(`/runs/${run}`)
  await page.getByRole('link', { name: 'Quality' }).click()
  await expect(page).toHaveURL(new RegExp(`/runs/${run}/quality$`))
})

test('the trust banner names the worst level and a card opens its section', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/quality`)
  await expect(page.locator('[data-role="trust"]')).toContainText('Needs attention')
  await expect(page.locator('[data-role="trust"]')).toContainText('fully clean')
  const section = page.locator('[data-section="usability"]')
  if (await section.evaluate((e: HTMLDetailsElement) => e.open)) {
    await section.locator('summary').click()
  }
  await expect(section).not.toHaveAttribute('open', '')
  await card(page, 'quality-usability').click()
  await expect(section).toHaveAttribute('open', '')
})

test('a finding row expands to its evidence and guidance', async ({ page, request }) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/quality`)
  const completeness = page.locator('[data-section="completeness"]')
  const row = completeness.locator('tr.row', { hasText: 'notes' }).first()
  await row.getByRole('button').click()
  const detail = completeness.locator('[data-role="finding-detail"]').first()
  await expect(detail).toContainText('null share')
  await expect(detail).toContainText('0.75')
  await expect(detail.locator('.guidance li').first()).toBeVisible()
  await row.getByRole('button').click()
  await expect(completeness.locator('[data-role="finding-detail"]')).toHaveCount(0)
})

test('the findings table sorts by column', async ({ page, request }) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/quality`)
  const completeness = page.locator('[data-section="completeness"]')
  await completeness.getByRole('button', { name: 'Column' }).click()
  await expect(completeness.locator('th[aria-sort="ascending"]')).toContainText('Column')
  const columns = await completeness.locator('tr.row td:nth-child(3)').allTextContents()
  expect(columns).toEqual([...columns].sort((a, b) => a.localeCompare(b)))
})

test('missingness that depends on another column is drawn as bars', async ({ page, request }) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/quality`)
  const device = page.locator('[data-column="device_type"]')
  await expect(device).toContainText('Depends on other columns')
  await expect(device.locator('[data-predictor="channel"]')).toContainText('0.89')
  await expect(page.locator('[data-column="notes"]')).toContainText('No dependence found')
})

test('near-duplicate rows are listed with their similarity', async ({ page, request }) => {
  const run = await runOf(request, 'cities')
  await page.goto(`/runs/${run}/quality`)
  const pairs = page.locator('[data-role="pair"]')
  await expect(pairs.first()).toContainText('%')
  await expect(pairs.first()).toContainText('name')
})

// With SCREENSHOT_DIR set, save the tab in both themes for review.
test('save the quality screenshots', async ({ page, request }) => {
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1280, height: 900 })
  for (const dataset of ['titanic', 'high_missingness']) {
    const run = await runOf(request, dataset)
    for (const theme of ['dark', 'light']) {
      await page.goto(`/runs/${run}/quality`)
      if (theme === 'light') await page.getByRole('button', { name: /theme/ }).click()
      await expect(page.locator('[data-role="trust"]')).toBeVisible()
      await page.screenshot({ path: `${dir}/quality-${dataset}-${theme}.png`, fullPage: true })
    }
  }
})

test('the tiny, near-clean and comprehensive archetypes read at their own levels', async ({
  page,
  request,
}) => {
  const expectations: Record<string, { trust: string; levels: Record<string, string> }> = {
    tiny: { trust: 'Worth investigating', levels: { 'quality-usability': 'Warning' } },
    near_clean: { trust: 'Minor issues', levels: { 'quality-completeness': 'Minor' } },
    comprehensive: {
      trust: 'Needs attention',
      levels: { 'quality-completeness': 'Serious', 'quality-redundancy': 'Warning' },
    },
  }
  for (const [dataset, { trust, levels }] of Object.entries(expectations)) {
    const run = await runOf(request, dataset)
    await page.goto(`/runs/${run}/quality`)
    await expect(page.locator('[data-role="trust"]')).toContainText(trust)
    for (const [id, text] of Object.entries(levels)) await expect(level(page, id)).toHaveText(text)
    await expect(level(page, 'quality-leakage')).toHaveText('Not assessed')
  }
})

test('a run without the near-duplicate check says why the card is empty', async ({
  page,
  request,
}) => {
  await page.goto(`/runs/${await runOf(request, 'clean')}/quality`)
  const card = page.locator('section', {
    has: page.getByRole('heading', { name: 'Near-duplicate rows' }),
  })
  await expect(card.locator('[data-role="check-missing"]')).toContainText('optional')
  await page.goto(`/runs/${await runOf(request, 'cities')}/quality`)
  await expect(card.locator('[data-role="pair"]').first()).toBeVisible()
})

test('a check that does not apply says it was not calculated, in plain words', async ({
  page,
  request,
}) => {
  await page.goto(`/runs/${await runOf(request, 'tiny')}/quality`)
  await expect(page.getByRole('heading', { name: 'Why values are missing' })).toBeVisible()
  await expect(page.getByText('Not calculated: this check does not apply')).toBeVisible()
  await expect(page.getByText('Status not_applicable')).toHaveCount(0)
})
