import { expect, test, type APIRequestContext } from '@playwright/test'

interface Summary {
  run_id: string
  dataset_key: string
}

async function runOf(request: APIRequestContext, dataset: string, sampled = false) {
  const { items } = (await (await request.get('/v2/runs')).json()) as { items: Summary[] }
  for (const { run_id, dataset_key } of items) {
    if (dataset_key !== dataset) continue
    const { header } = (await (await request.get(`/v2/runs/${run_id}`)).json()) as {
      header: { dataset: { sampling: unknown } }
    }
    if ((header.dataset.sampling !== null) === sampled) return run_id
  }
  throw new Error(`the fixture store has no ${dataset} run`)
}

test('filters live in the URL and the count matches the API', async ({ page, request }) => {
  const run = await runOf(request, 'titanic')
  const expected = (await (
    await request.get(`/v2/runs/${run}/findings?min_severity=warn&limit=1`)
  ).json()) as { total: number }

  await page.goto(`/runs/${run}/findings`)
  await page.getByLabel('Severity at least').selectOption('warn')
  await page.getByRole('button', { name: 'Apply' }).click()

  await expect(page).toHaveURL(/severity=warn/)
  await expect(page.getByRole('status')).toHaveText(`${expected.total} findings`)

  await page.reload()
  await expect(page.getByLabel('Severity at least')).toHaveValue('warn')
  await expect(page.getByRole('status')).toHaveText(`${expected.total} findings`)
})

test('opening a finding shows its evidence, and a top finding links here', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}`)
  await page.getByRole('region', { name: 'Top findings' }).getByRole('link').first().click()

  await expect(page).toHaveURL(/findings\?finding=/)
  const detail = page.getByRole('region', { name: /./, exact: false }).filter({
    has: page.getByRole('heading', { name: 'Evidence' }),
  })
  await expect(detail).toContainText('rows')
  await page.getByRole('button', { name: 'Close' }).click()
  await expect(page).not.toHaveURL(/finding=/)
})

test('a fuzzy-duplicate finding offers its pair list, paged on demand', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'cities')
  const artifactRequests: string[] = []
  page.on('request', (req) => {
    if (req.url().includes('/artifacts/')) artifactRequests.push(req.url())
  })

  await page.goto(`/runs/${run}/findings?code=near_duplicate_rows`)
  await page.getByRole('link', { name: /near-duplicate rows/ }).click()
  await expect(page.getByRole('heading', { name: 'Full detail' })).toBeVisible()
  expect(artifactRequests).toHaveLength(0)

  await page.getByRole('button', { name: /^Load / }).click()
  await expect(page.getByText('1–20 of 28')).toBeVisible()
  await page.getByRole('button', { name: 'Next' }).click()
  await expect(page.getByText('21–28 of 28')).toBeVisible()
  expect(artifactRequests).toHaveLength(2)
  expect(artifactRequests[1]).toContain('offset=20')
})

test('the count chart filters the explorer, and the Overview counts link here', async ({
  page,
  request,
}) => {
  const run = await runOf(request, 'titanic')
  await page.goto(`/runs/${run}`)
  await page
    .getByRole('region', { name: 'Findings by dimension and severity' })
    .getByRole('link')
    .first()
    .click()
  await expect(page).toHaveURL(/findings\?dimension=/)
  const dimension = new URL(page.url()).searchParams.get('dimension') ?? ''
  const expected = (await (
    await request.get(`/v2/runs/${run}/findings?dimension=${dimension}&limit=1`)
  ).json()) as { total: number }
  await expect(page.getByRole('status')).toHaveText(`${expected.total} findings`)
  await expect(page.getByRole('combobox', { name: 'Dimension' })).toHaveValue(dimension)
})
