import { expect, test, type APIRequestContext } from '@playwright/test'

interface Summary {
  run_id: string
  dataset_key: string
}

async function runsOf(request: APIRequestContext, dataset: string) {
  const { items } = (await (await request.get('/v2/runs')).json()) as { items: Summary[] }
  return items.filter((run) => run.dataset_key === dataset).map((run) => run.run_id)
}

// The runs the review looks at, with the tabs worth seeing on each: titanic for the usual
// dataset, then the archetype that stresses each tab.
const REVIEW: Record<string, string[]> = {
  titanic: [
    '',
    'quality',
    'distributions',
    'columns',
    'columns/age',
    'columns/sex',
    'relationships',
    'ml-readiness',
    'time-series',
    'findings',
  ],
  high_missingness: ['', 'quality', 'ml-readiness'],
  severe_multicollinearity: ['relationships'],
  wide: ['', 'relationships'],
  all_categorical: ['relationships', 'columns'],
  single_column: ['', 'relationships'],
  clean: ['', 'quality', 'ml-readiness'],
  near_clean: ['quality', 'ml-readiness'],
  tiny: ['', 'quality'],
  comprehensive: ['', 'quality', 'distributions', 'ml-readiness'],
  kitchen5k: ['time-series'],
  cities: ['quality'],
}

const slug = (tab: string) => tab.replaceAll('/', '-') || 'overview'

// With SCREENSHOT_DIR set, save every page of the review in both themes: one full-page
// image per page, named `<dataset>-<page>-<theme>.png`. Without it the test is skipped.
test('save a screenshot of every page in both themes', async ({ page, request }) => {
  test.setTimeout(10 * 60_000)
  const dir = process.env.SCREENSHOT_DIR
  test.skip(!dir, 'SCREENSHOT_DIR is not set')
  await page.setViewportSize({ width: 1280, height: 800 })

  const shoot = async (url: string, name: string, ready: string) => {
    for (const theme of ['dark', 'light']) {
      await page.goto(url)
      if (theme === 'light') await page.getByRole('button', { name: /theme/ }).click()
      await expect(page.locator(ready).first()).toBeVisible()
      await page.waitForTimeout(400) // charts finish their entrance animation
      await page.screenshot({ path: `${dir}/${name}-${theme}.png`, fullPage: true })
    }
  }

  await shoot('/', 'datasets', '[data-dataset="titanic"]')
  await shoot('/datasets/titanic', 'dataset-titanic', '[data-section="settings"]')
  for (const [dataset, tabs] of Object.entries(REVIEW)) {
    const [run] = await runsOf(request, dataset)
    if (!run) throw new Error(`the fixture store has no ${dataset} run`)
    for (const tab of tabs)
      await shoot(`/runs/${run}/${tab}`, `${dataset}-${slug(tab)}`, '[data-role="health-bar"]')
  }
  const titanic = await runsOf(request, 'titanic')
  if (titanic.length === 2)
    await shoot(
      `/compare?base=${titanic[1]}&head=${titanic[0]}`,
      'compare-titanic',
      '[data-role="diff-summary"]',
    )
})
