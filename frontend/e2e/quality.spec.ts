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
  await expect(
    page.getByRole('heading', { name: 'Quality and ML readiness', level: 1 }),
  ).toBeVisible()
  await expect(page.locator('[data-role="gate"]')).toContainText('Ready for modelling')
  await expect(page.locator('[data-role="caveat"]')).toContainText('Leakage risk')

  for (const id of [
    'quality-completeness',
    'quality-validity',
    'quality-usability',
    'quality-redundancy',
    'ml-transformations',
    'ml-encoding',
    'ml-missingness',
    'ml-unusable',
  ]) {
    await expect(level(page, id)).toHaveText('All clear')
  }
  for (const id of ['quality-leakage', 'ml-leakage']) {
    await expect(level(page, id)).toHaveText('Not assessed')
    await expect(card(page, id).locator('[data-role="not-assessed"]')).toBeVisible()
    await expect(level(page, id)).not.toHaveClass(/badge-ok/)
  }
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
  await page.goto(`/runs/${run}/quality`)
  await expect(page.locator('[data-role="gate"]')).toContainText('Not ready for modelling')
  await expect(level(page, 'quality-completeness')).toHaveText('Serious')
  await expect(level(page, 'ml-missingness')).toHaveText('Serious')
  const completeness = card(page, 'quality-completeness')
  for (const column of ['device_type', 'premium_score', 'notes']) {
    await expect(completeness).toContainText(column)
  }
})

test('a readiness finding offers its actions and links to the findings explorer', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/quality`)
  const missingness = card(page, 'ml-missingness')
  await expect(missingness.locator('.actions li').first()).toBeVisible()
  await missingness.locator('.findings a').first().click()
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
