import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      // Proxy all /api requests to FastAPI during development
      // This means the Vue frontend never makes cross-origin requests
      // the browser always talks to localhost:5173 and Vite forwards
      // the request to the FastAPI server. No CORS issues in dev.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
