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

const dimension = (page: Page, name: string) => page.locator(`[data-dimension="${name}"]`)

test('every tab shows the header, the health bar and the six tabs', async ({ page, request }) => {
  const run = await runOf(request, 'high_missingness')
  for (const path of ['', '/quality', '/distributions', '/relationships', '/ml-readiness']) {
    await page.goto(`/runs/${run}${path}`)
    await expect(page.locator('[data-role="breadcrumb"]')).toContainText('high_missingness')
    await expect(page.locator('[data-role="health-bar"] li')).toHaveCount(5)
    await expect(page.getByRole('navigation', { name: 'Run pages' }).getByRole('link')).toHaveText([
      'Overview',
      'Quality',
      'Distributions',
      'Relationships',
      'ML Readiness',
      'Time Series',
    ])
  }
})

test('the health bar reports a serious dimension with its column count', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'high_missingness')
  await page.goto(`/runs/${run}`)

  const completeness = dimension(page, 'completeness')
  await expect(completeness).toHaveAttribute('data-mark', 'red')
  await expect(completeness.locator('[data-role="summary"]')).toHaveText(/^\d+ cols?$/)
  await expect(dimension(page, 'redundancy')).toHaveAttribute('data-mark', 'green')
  await expect(dimension(page, 'redundancy').locator('[data-role="summary"]')).toHaveText(
    'All clear',
  )
})

test('a dimension without a target is grey and says it is not assessed', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}`)

  const leakage = dimension(page, 'leakage')
  await expect(leakage).toHaveAttribute('data-mark', 'none')
  await expect(leakage.locator('[data-role="summary"]')).toHaveText('Not assessed')
  await leakage.hover()
  await expect(leakage.getByRole('tooltip')).toContainText('Not assessed')
})

test('the health bar dims the dimensions a tab does not speak to', async ({ page, request }) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/relationships`)

  await expect(dimension(page, 'completeness')).toHaveClass(/dimmed/)
  await expect(dimension(page, 'redundancy')).not.toHaveClass(/dimmed/)
  await expect(dimension(page, 'leakage')).not.toHaveClass(/dimmed/)
})

test('the tab bar follows the route, and the Phase 6 column address redirects', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  const tab = (name: string) => page.getByRole('link', { name, exact: true })

  await page.goto(`/runs/${run}`)
  await expect(tab('Overview')).toHaveAttribute('aria-current', 'page')

  await tab('Quality').click()
  await expect(page).toHaveURL(`/runs/${run}/quality`)
  await expect(tab('Quality')).toHaveAttribute('aria-current', 'page')
  await expect(tab('Overview')).not.toHaveAttribute('aria-current', 'page')

  await page.goto(`/runs/${run}/columns`)
  await expect(page).toHaveURL(`/runs/${run}/distributions`)
  await expect(tab('Distributions')).toHaveAttribute('aria-current', 'page')

  await page.goto(`/runs/${run}/findings`)
  await expect(page.locator('[data-tab][aria-current="page"]')).toHaveCount(0)
})

test('the shell survives a reload on a deep link', async ({ page, request }) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}/ml-readiness`)
  await page.reload()
  await expect(page.getByRole('heading', { name: 'ML readiness', level: 1 })).toBeVisible()
  await expect(page.locator('[data-role="health-bar"] li')).toHaveCount(5)
})

// With SCREENSHOT_DIR set, save the shell in both themes for review.
test('save the shell screenshots', async ({ page, request }) => {
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1280, height: 520 })
  for (const dataset of ['titanic', 'high_missingness']) {
    const run = await runOf(request, dataset)
    for (const theme of ['dark', 'light']) {
      await page.goto(`/runs/${run}`)
      const toggle = page.getByRole('button', { name: /theme/ })
      if (theme === 'light') await toggle.click()
      await expect(page.locator('[data-role="health-bar"] li')).toHaveCount(5)
      await page.screenshot({ path: `${dir}/shell-${dataset}-${theme}.png` })
    }
  }
})
