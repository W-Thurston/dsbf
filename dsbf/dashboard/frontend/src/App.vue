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
// App shell - just the header and router outlet.
// All page logic lives in the view components.
</script>

<style>
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
  box-sizing: border-box;
}

/* Allows cards in flex rows to stretch to equal height */
.row-split > .card,
.row-equal > .card,
.row-sample .card {
  height: 100%;
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

/* ── Light theme overrides ─────────────────────────────────────────────────── */
body.theme-light {
  background: #f1f5f9;
  color: #1e293b;
}

body.theme-light a { color: #2563eb; }

body.theme-light .app-header {
  background: #ffffff;
  border-bottom-color: #e2e8f0;
}

body.theme-light .breadcrumb       { color: #64748b; }
body.theme-light .breadcrumb a     { color: #2563eb; }
body.theme-light .breadcrumb .sep  { color: #94a3b8; }

body.theme-light .card {
  background: #ffffff;
  border-color: #e2e8f0;
  color: #1e293b;
}

/* Tab bar */
body.theme-light .tab-bar          { border-bottom-color: #e2e8f0; }
body.theme-light .tab-btn          { color: #64748b; }
body.theme-light .tab-btn:hover    { color: #1e293b; }
body.theme-light .tab-btn.active   { color: #2563eb; border-bottom-color: #2563eb; }

/* Theme toggle buttons */
body.theme-light .theme-btn        { background: #f8fafc; border-color: #e2e8f0; color: #64748b; }
body.theme-light .theme-btn.active { background: #dbeafe; border-color: #2563eb; color: #2563eb; }

/* Meta strip */
body.theme-light .meta-metric      { border-right-color: #e2e8f0; }
body.theme-light .meta-label       { color: #94a3b8; }
body.theme-light .meta-value       { color: #1e293b; }

/* Quality header */
body.theme-light .quality-title    { color: #64748b; }
body.theme-light .quality-divider  { background: #e2e8f0; }
body.theme-light .category-label   { color: #94a3b8; }
body.theme-light .category-value   { color: #1e293b; }

/* Tables */
body.theme-light .meta-table th   { background: #f8fafc; color: #94a3b8; border-bottom-color: #e2e8f0; }
body.theme-light .meta-table td   { color: #1e293b; border-bottom-color: #f1f5f9; }
body.theme-light .meta-table tr:hover td { background: #f1f5f9; }
body.theme-light td.col-frozen     { background: #f8fafc !important; border-right-color: #e2e8f0; }
body.theme-light .meta-table tr:hover td.col-frozen { background: #f1f5f9 !important; }
body.theme-light .col-name        { color: #0f172a; }
body.theme-light .muted           { color: #64748b; }

/* Search input */
body.theme-light .search-input     { background: #f8fafc; border-color: #e2e8f0; color: #1e293b; }
body.theme-light .search-input:focus { border-color: #2563eb; }

/* Alerts */
body.theme-light .alert-error { background: #fef2f2; border-color: #f87171; color: #991b1b; }
body.theme-light .alert-warn  { background: #fff7ed; border-color: #fb923c; color: #9a3412; }
body.theme-light .alert-info  { background: #eff6ff; border-color: #60a5fa; color: #1d4ed8; }
body.theme-light .no-alerts   { color: #16a34a; }

/* Runs table */
body.theme-light .runs-table th { color: #94a3b8; border-bottom-color: #e2e8f0; }
body.theme-light .runs-table td { color: #1e293b; border-bottom-color: #f1f5f9; }
body.theme-light .run-row:hover td { background: #f1f5f9; }
body.theme-light .run-key       { color: #2563eb; }

/* Dataset cards */
body.theme-light .dataset-card:hover { border-color: #2563eb; }
body.theme-light .ds-name            { color: #0f172a; }
body.theme-light .meta-label         { color: #94a3b8; }

/* Dataset sample table */
body.theme-light .sample-table th              { background: #f8fafc; color: #94a3b8; border-bottom-color: #e2e8f0; }
body.theme-light .sample-table td              { color: #1e293b; border-bottom-color: #f1f5f9; }
body.theme-light .sample-table tr:hover td     { background: #f1f5f9; }
body.theme-light .sample-table .row-idx        { color: #94a3b8 !important; }
body.theme-light .sample-table .null-cell      { color: #94a3b8 !important; }
body.theme-light .n-select                     { background: #f8fafc; border-color: #e2e8f0; color: #1e293b; }
body.theme-light .n-label                      { color: #64748b; }

/* Run information card */
body.theme-light .run-date-value               { color: #1e293b; }
body.theme-light .run-date-label               { color: #64748b; }
body.theme-light .run-date-item                { border-bottom-color: #e2e8f0; }

/* Dataset lineage card */
body.theme-light .lineage-value                { color: #1e293b; }
body.theme-light .lineage-label                { color: #64748b; }
body.theme-light .lineage-item                 { border-bottom-color: #e2e8f0; }
body.theme-light .lineage-note                 { background: #f8fafc; border-color: #e2e8f0; color: #64748b; }
</style>
