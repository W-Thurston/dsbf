import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  test: {
    // The unit tests cover pure functions and the client, so they need no DOM.
    environment: 'node',
    include: ['src/**/*.test.ts'], // e2e/ is Playwright's
  },
  server: {
    proxy: {
      // Forward /v2 to `dsbf dashboard --no-open` so the browser only ever talks to
      // localhost:5173 and no CORS is needed in development.
      '/v2': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
    },
  },
})
