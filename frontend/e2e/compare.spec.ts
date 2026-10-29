import { expect, test, type APIRequestContext } from '@playwright/test'

interface Summary {
  run_id: string
  dataset_key: string
  started_at: string
}

async function runsOf(request: APIRequestContext, dataset: string) {
  const { items } = (await (await request.get('/v2/runs')).json()) as { items: Summary[] }
  return items.filter((run) => run.dataset_key === dataset) // newest first
}

test('a run links to a comparison with its previous run, and the diff shows the planted change', async ({
  page,
  request,
}) => {
  const [head, base] = await runsOf(request, 'titanic')
  await page.goto(`/runs/${head!.run_id}`)
  await page.locator('[data-role="compare-link"]').click()

  await expect(page.getByRole('heading', { name: 'Compare runs' })).toBeVisible()
  await expect(page.locator('[data-role="base"]')).toHaveValue(base!.run_id)
  await expect(page.locator('[data-role="head"]')).toHaveValue(head!.run_id)
  await expect(page.locator('[data-role="diff-summary"]')).toBeVisible()
  await expect(page.locator('[data-role="different-datasets"]')).toHaveCount(0)
  // The head run is the 300-row sample, which also ran a check that raises.
  await expect(page.locator('[data-section="status-changes"] tbody tr').first()).toBeVisible()
  await expect(page.locator('[data-section="status-changes"]')).toContainText('always_fails')
})

test('runs of two datasets compare and the view says they differ', async ({ page, request }) => {
  const [titanic] = await runsOf(request, 'titanic')
  const [cities] = await runsOf(request, 'cities')
  await page.goto(`/compare?base=${titanic!.run_id}&head=${cities!.run_id}`)

  await expect(page.locator('[data-role="different-datasets"]')).toContainText('not of the same')
  await expect(page.locator('[data-role="diff-summary"]')).toBeVisible()
})

test('the compare view asks for two runs before it compares', async ({ page }) => {
  await page.goto('/compare')
  await expect(page.locator('[data-role="compare-prompt"]')).toBeVisible()
})

test('the dataset page flags a changed source location in the lineage', async ({ page }) => {
  await page.goto('/datasets/relocated')

  const rows = page.locator('[data-section="lineage"] tbody tr')
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0).locator('[data-flag="location"]')).toBeVisible() // the newest
  await expect(rows.nth(0).locator('[data-flag="fingerprint"]')).toHaveCount(0)
  await expect(rows.nth(1).locator('[data-flag]')).toHaveCount(0)
})

test('the dataset page follows one finding across the dataset runs', async ({ page }) => {
  await page.goto('/datasets/titanic')

  const history = page.locator('[data-section="history"]')
  await expect(history.locator('tbody tr').first()).toBeVisible()
  expect(await history.locator('[data-role="history-finding"] option').count()).toBeGreaterThan(0)
})
