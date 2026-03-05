<template>
  <div id="app">
    <header class="app-header">
      <nav class="breadcrumb">
        <router-link to="/">DSBF</router-link>
        <template v-if="$route.params.name">
          <span class="sep">›</span>
          <router-link :to="`/datasets/${$route.params.name}`">
            {{ $route.params.name }}
          </router-link>
        </template>
        <template v-if="$route.params.runKey">
          <span class="sep">›</span>
          <span>{{ $route.params.runKey }}</span>
        </template>
      </nav>
    </header>

    <main>
      <router-view />
    </main>
  </div>
</template>

<script setup>
// App shell — just the header and router outlet.
// All page logic lives in the view components.
</script>

<style>
/* ── Reset & base ──────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  line-height: 1.6;
  min-height: 100vh;
}

a { color: #60a5fa; text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── App header ────────────────────────────────────────────────────────────── */
.app-header {
  background: #1e293b;
  border-bottom: 1px solid #334155;
  padding: 12px 24px;
}

.breadcrumb {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  color: #94a3b8;
}

.breadcrumb a { color: #60a5fa; }
.breadcrumb .sep { color: #475569; }

main {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}

/* ── Shared component styles ───────────────────────────────────────────────── */
.card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 20px;
}

.card-title {
  font-size: 13px;
  font-weight: 600;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 12px;
}

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 500;
}

.badge-blue   { background: #1e3a5f; color: #60a5fa; border: 1px solid #60a5fa; }
.badge-green  { background: #14291f; color: #4ade80; border: 1px solid #4ade80; }
.badge-orange { background: #3d2510; color: #fb923c; border: 1px solid #fb923c; }
.badge-red    { background: #3d0f0f; color: #f87171; border: 1px solid #f87171; }
.badge-gray   { background: #1e293b; color: #94a3b8; border: 1px solid #475569; }

.loading {
  color: #94a3b8;
  padding: 40px;
  text-align: center;
  font-size: 14px;
}

.error {
  color: #f87171;
  padding: 20px;
  background: #3d0f0f;
  border: 1px solid #f87171;
  border-radius: 8px;
  font-size: 14px;
}
</style>
