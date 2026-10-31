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

test('the gate banner names the verdict and the dimension cards name their levels', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/ml-readiness`)
  await expect(page.locator('[data-role="gate"]')).toContainText('Not ready for modelling')
  await expect(page.locator('[data-dimension^="ml-"]')).toHaveCount(5)
  await expect(page.locator('[data-dimension="ml-missingness"] [data-role="level"]')).toHaveText(
    'Serious',
  )
})

test('a card opens its section', async ({ page, request }) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/ml-readiness`)
  const section = page.locator('[data-section="encoding"]')
  if (await section.evaluate((e: HTMLDetailsElement) => e.open)) {
    await section.locator('summary').click()
  }
  await expect(section).not.toHaveAttribute('open', '')
  await page.locator('[data-dimension="ml-encoding"]').click()
  await expect(section).toHaveAttribute('open', '')
})

test('a suggested action explains what it does, its trade-off and what follows', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}/ml-readiness`)
  const missingness = page.locator('[data-section="missingness"]')
  await missingness.locator('tr.row').first().getByRole('button').click()
  const note = missingness.locator('[data-role="action-note"]').first()
  await expect(note).toContainText('What it does')
  await expect(note).toContainText('Trade-off')
  await expect(note).toContainText('After this')
})

test('the transformation preview charts skewness before and after for a skewed column', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  const output = (await (await request.get(`/v2/runs/${run}/checks/transformations`)).json()) as {
    output: { columns: Record<string, { recommended: string | null }> }
  }
  const [column, ...rest] = Object.keys(output.output.columns)
  expect(column, 'titanic has a skewed column').toBeTruthy()
  void rest
  await page.goto(`/runs/${run}/ml-readiness`)
  const preview = page.locator(`[data-role="transform-preview"] [data-column="${column}"]`)
  await expect(preview.locator('[data-bar="before"]')).toBeVisible()
  const recommended = output.output.columns[column!]!.recommended
  if (recommended) await expect(preview.locator(`[data-bar="${recommended} ★"]`)).toBeVisible()
})

test('ready features list the columns with no readiness finding', async ({ page, request }) => {
  const run = await runOf(request, 'clean')
  const ml = (await (await request.get(`/v2/runs/${run}/checks/ml_readiness`)).json()) as {
    output: { clean_columns: string[] }
  }
  await page.goto(`/runs/${run}/ml-readiness`)
  await expect(page.locator('[data-role="ready-features"] li')).toHaveCount(
    ml.output.clean_columns.length,
  )
})

// With SCREENSHOT_DIR set, save the tab in both themes for review.
test('save the ML readiness screenshots', async ({ page, request }) => {
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1280, height: 900 })
  for (const dataset of ['titanic', 'high_missingness']) {
    const run = await runOf(request, dataset)
    for (const theme of ['dark', 'light']) {
      await page.goto(`/runs/${run}/ml-readiness`)
      if (theme === 'light') await page.getByRole('button', { name: /theme/ }).click()
      await expect(page.locator('[data-role="gate"]')).toBeVisible()
      await expect(page.locator('[data-role="transform-preview"]')).toBeVisible()
      await page.screenshot({ path: `${dir}/ml-readiness-${dataset}-${theme}.png`, fullPage: true })
    }
  }
})

test('the gate follows the archetype: tiny is ready, near-clean needs work, comprehensive is not ready', async ({
  page,
  request,
}) => {
  const verdicts = {
    tiny: 'Ready for modelling',
    near_clean: 'Needs work before modelling',
    comprehensive: 'Not ready for modelling',
  }
  for (const [dataset, verdict] of Object.entries(verdicts)) {
    await page.goto(`/runs/${await runOf(request, dataset)}/ml-readiness`)
    await expect(page.locator('[data-role="gate"]')).toContainText(verdict)
    await expect(page.locator('[data-dimension="ml-leakage"] [data-role="level"]')).toHaveText(
      'Not assessed',
    )
  }
})
