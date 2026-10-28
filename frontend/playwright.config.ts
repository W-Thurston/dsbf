import { defineConfig } from '@playwright/test'

const PORT = 8799

// The dashboard is served by `dsbf dashboard` from a store built by e2e/fixture_store.py,
// so the tests exercise the same bundle and API a user gets. Run `pnpm build` first.
export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? 'github' : 'list',
  use: { baseURL: `http://127.0.0.1:${PORT}` },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
  webServer: {
    command:
      'rm -rf .e2e-home && uv run python e2e/fixture_store.py .e2e-home && ' +
      `DSBF_HOME=.e2e-home uv run dsbf dashboard --no-open --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/v2/datasets`,
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
  },
})
