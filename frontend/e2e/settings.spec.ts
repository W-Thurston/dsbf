import { expect, test } from '@playwright/test'

// One worker takes the jobs in turn, and a standard titanic run takes seconds.
const JOB = { timeout: 90_000 }

test('saving a target and rerunning lands on a run in which class_imbalance ran', async ({
  page,
  request,
}) => {
  await page.goto('/datasets/settings')
  const form = page.locator('[data-section="settings"]')
  await expect(form.locator('[data-field="depth"]')).toBeVisible()
  // The dataset key is not a setting.
  await expect(form.locator('[data-field="dataset"]')).toHaveCount(0)

  await form.locator('[data-field="target"]').fill('survived')
  await form.locator('[data-role="rerun"]').click()

  await expect(page).toHaveURL(/\/runs\//, JOB)
  const runId = page.url().split('/runs/')[1]!
  const result = await (await request.get(`/v2/runs/${runId}/checks/class_imbalance`)).json()
  expect(result.status).toBe('ok')

  // The overlay was saved, so it is there when the dataset page is opened again.
  await page.goto('/datasets/settings')
  await expect(page.locator('[data-field="target"]')).toHaveValue('survived')
  await page.locator('[data-role="reset"]').click()
  await expect(page.locator('[data-field="target"]')).toHaveValue('')
})

test('a rerun of a source that no longer exists fails and shows why', async ({ page }) => {
  await page.goto('/datasets/relocated')
  await page.locator('[data-role="rerun"]').click()
  await expect(page.locator('[data-role="job-state"]')).toHaveText('failed', JOB)
  await expect(page.locator('[data-role="job-error"]')).toContainText('titanic_moved.csv')
})

// With SCREENSHOT_DIR set, save the settings form and a rerun in progress in both themes.
test('save the settings screenshots', async ({ page }) => {
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1280, height: 900 })
  const form = page.locator('[data-section="settings"]')
  for (const theme of ['dark', 'light']) {
    await page.goto('/datasets/titanic')
    if (theme === 'light') await page.getByRole('button', { name: /theme/ }).click()
    await expect(form.locator('[data-field="depth"]')).toBeVisible()
    await form.screenshot({ path: `${dir}/settings-${theme}.png` })
    await form.locator('[data-role="rerun"]').click()
    await expect(page.locator('[data-role="rerun-progress"]')).toBeVisible()
    await form.screenshot({ path: `${dir}/rerun-progress-${theme}.png` })
    await expect(page).toHaveURL(/\/runs\//, JOB)
  }
})
