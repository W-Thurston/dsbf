# DSBF dashboard

Vue 3 + Vite single-page app that reads from the DSBF API. It uses pnpm; run `corepack enable`
once so the version in `package.json` (`packageManager`) is fetched.

```bash
pnpm install
pnpm dev         # http://localhost:5173, proxies /api to the API on :8000
pnpm lint        # ESLint, then a Prettier check
pnpm build
pnpm test:run    # Vitest
```

See [`docs/architecture.md`](../docs/architecture.md) for how the dashboard fits with the rest of DSBF.
