import { expect, test } from '@playwright/test'

test('the dataset list is a card grid with five dimension dots per dataset', async ({ page }) => {
  await page.goto('/')
  const titanic = page.locator('[data-dataset="titanic"]')
  await expect(titanic).toBeVisible()
  await expect(titanic.locator('[data-dimension]')).toHaveCount(5)
  await expect(titanic).toContainText('Runs')
})

test('searching narrows the cards and says when nothing matches', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-dataset="cities"]')).toBeVisible()
  await page.locator('[data-role="search"]').fill('tita')
  await expect(page.locator('[data-dataset="titanic"]')).toBeVisible()
  await expect(page.locator('[data-dataset="cities"]')).toHaveCount(0)
  await page.locator('[data-role="search"]').fill('zzzz')
  await expect(page.locator('[data-role="no-match"]')).toBeVisible()
})

test('a card opens the dataset page', async ({ page }) => {
  await page.goto('/')
  await page.locator('[data-dataset="titanic"]').click()
  await expect(page).toHaveURL(/\/datasets\/titanic/)
})

test('the compare view charts the count of each kind of change', async ({ page, request }) => {
  const { items } = (await (await request.get('/v2/runs')).json()) as {
    items: { run_id: string; dataset_key: string }[]
  }
  const [head, base] = items.filter((run) => run.dataset_key === 'titanic')
  await page.goto(`/compare?base=${base!.run_id}&head=${head!.run_id}`)
  const bars = page.locator('[data-section="diff-counts"] [data-bar]')
  await expect(bars).toHaveCount(4)
})

// With SCREENSHOT_DIR set, save the dataset-level pages in both themes for review.
test('save the dataset page screenshots', async ({ page, request }) => {
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1280, height: 900 })
  const { items } = (await (await request.get('/v2/runs')).json()) as {
    items: { run_id: string; dataset_key: string }[]
  }
  const [head, base] = items.filter((run) => run.dataset_key === 'titanic')
  const pages = [
    ['dataset-list', '/', '[data-section="dataset-cards"]'],
    ['run-list', '/datasets/titanic', '[data-section="history"] table'],
    [
      'compare',
      `/compare?base=${base!.run_id}&head=${head!.run_id}`,
      '[data-section="diff-counts"]',
    ],
  ] as const
  for (const [name, path, ready] of pages) {
    for (const theme of ['dark', 'light']) {
      await page.goto(path)
      if (theme === 'light') await page.getByRole('button', { name: /theme/ }).click()
      await expect(page.locator(ready)).toBeVisible()
      await page.screenshot({ path: `${dir}/${name}-${theme}.png`, fullPage: true })
    }
  }
})
