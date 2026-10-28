# DSBF dashboard

TypeScript Vue 3 + Vite single-page app that reads from the DSBF API (`dsbf dashboard`) through a client generated from `schema/openapi.json`. It uses pnpm; run `corepack enable`
once so the version in `package.json` (`packageManager`) is fetched.

```bash
pnpm install
pnpm dev         # http://localhost:5173, proxies /v2 to `dsbf dashboard --no-open` on :8765
pnpm lint        # ESLint, then a Prettier check
pnpm build       # vue-tsc, then Vite
pnpm test:run    # Vitest
```

See [`docs/architecture.md`](../docs/architecture.md) for how the dashboard fits with the rest of DSBF.
