// dsbf/dashboard/frontend/src/chartTheme.js
//
// ECharts theme objects for DSBF dark and light modes.
// Colors are taken verbatim from style.css and App.vue to guarantee
// charts always match the surrounding dashboard UI.
//
// Usage:
//   import { registerDsbfThemes } from './chartTheme.js'
//   registerDsbfThemes()   // call once in main.js
//
//   Then pass theme="dsbf-dark" or theme="dsbf-light" to <v-chart>.

import * as echarts from 'echarts/core'

// ── Shared palette ────────────────────────────────────────────────────────────
// Data series colors — chosen to be distinguishable and consistent with the
// badge/badge-blue/green/orange/red vocabulary used across the dashboard.
const PALETTE = [
  '#60a5fa', // blue   (primary)
  '#4ade80', // green
  '#fb923c', // orange
  '#f87171', // red
  '#a78bfa', // purple (seasonal)
  '#34d399', // emerald
  '#fbbf24', // amber
  '#38bdf8', // sky
  '#e879f9', // fuchsia
  '#94a3b8', // slate  (fallback)
]

// ── Dark theme ────────────────────────────────────────────────────────────────
const DARK = {
  color: PALETTE,

  backgroundColor: 'transparent',   // card background comes from .card CSS

  textStyle: {
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif",
    color: '#94a3b8',
    fontSize: 12,
  },

  title: {
    textStyle:    { color: '#e2e8f0', fontSize: 13, fontWeight: 600 },
    subtextStyle: { color: '#64748b', fontSize: 11 },
  },

  legend: {
    textStyle: { color: '#94a3b8', fontSize: 11 },
    pageTextStyle: { color: '#64748b' },
    inactiveColor: '#334155',
  },

  tooltip: {
    backgroundColor: '#1e293b',
    borderColor:     '#334155',
    borderWidth:     1,
    textStyle:       { color: '#e2e8f0', fontSize: 12 },
    extraCssText:    'box-shadow: 0 4px 12px rgba(0,0,0,0.4); border-radius: 6px;',
  },

  axisPointer: {
    lineStyle:  { color: '#475569', width: 1 },
    crossStyle: { color: '#475569' },
    label:      { backgroundColor: '#1e293b', color: '#94a3b8', borderColor: '#334155' },
  },

  toolbox: {
    iconStyle:   { borderColor: '#64748b' },
    emphasis: {
      iconStyle: { borderColor: '#60a5fa' },
    },
  },

  dataZoom: [
    {
      backgroundColor:   '#0f172a',
      dataBackground: {
        lineStyle:  { color: '#334155' },
        areaStyle:  { color: '#1e293b' },
      },
      fillerColor:        'rgba(96, 165, 250, 0.1)',
      handleStyle:        { color: '#60a5fa', borderColor: '#3b82f6' },
      moveHandleStyle:    { color: '#60a5fa' },
      textStyle:          { color: '#64748b' },
      borderColor:        '#334155',
    },
  ],

  visualMap: {
    textStyle: { color: '#94a3b8' },
  },

  timeline: {
    lineStyle:    { color: '#334155' },
    itemStyle:    { color: '#60a5fa' },
    label:        { color: '#64748b' },
    controlStyle: { color: '#64748b', borderColor: '#64748b' },
  },

  // ── Axis shared defaults ──────────────────────────────────────────────────
  categoryAxis: {
    axisLine:   { lineStyle: { color: '#1e293b' } },
    axisTick:   { lineStyle: { color: '#1e293b' } },
    axisLabel:  { color: '#64748b', fontSize: 11 },
    splitLine:  { lineStyle: { color: '#1e293b', type: 'dashed' } },
    splitArea:  { areaStyle: { color: ['rgba(255,255,255,0.01)', 'transparent'] } },
  },

  valueAxis: {
    axisLine:   { lineStyle: { color: '#1e293b' } },
    axisTick:   { lineStyle: { color: '#1e293b' } },
    axisLabel:  { color: '#64748b', fontSize: 11 },
    splitLine:  { lineStyle: { color: '#1e293b', type: 'dashed' } },
    splitArea:  { areaStyle: { color: ['rgba(255,255,255,0.01)', 'transparent'] } },
  },

  logAxis: {
    axisLine:   { lineStyle: { color: '#1e293b' } },
    axisTick:   { lineStyle: { color: '#1e293b' } },
    axisLabel:  { color: '#64748b', fontSize: 11 },
    splitLine:  { lineStyle: { color: '#1e293b', type: 'dashed' } },
  },

  timeAxis: {
    axisLine:   { lineStyle: { color: '#1e293b' } },
    axisTick:   { lineStyle: { color: '#1e293b' } },
    axisLabel:  { color: '#64748b', fontSize: 11 },
    splitLine:  { lineStyle: { color: '#1e293b', type: 'dashed' } },
  },

  // ── Series shared defaults ────────────────────────────────────────────────
  line: {
    symbol:         'circle',
    symbolSize:     4,
    smooth:         false,
    lineStyle:      { width: 2 },
    itemStyle:      { borderWidth: 2 },
  },

  bar: {
    barMaxWidth:    32,
    itemStyle:      { borderRadius: [2, 2, 0, 0] },
  },

  scatter: {
    itemStyle: { opacity: 0.8 },
  },

  heatmap: {
    itemStyle: { borderColor: '#0f172a', borderWidth: 1 },
  },

  // ── Grid ──────────────────────────────────────────────────────────────────
  grid: {
    borderColor: '#1e293b',
  },
}

// ── Light theme ───────────────────────────────────────────────────────────────
// Derived from App.vue body.theme-light overrides.
const LIGHT = {
  ...DARK,

  backgroundColor: 'transparent',

  textStyle: { ...DARK.textStyle, color: '#64748b' },

  title: {
    textStyle:    { color: '#1e293b', fontSize: 13, fontWeight: 600 },
    subtextStyle: { color: '#94a3b8', fontSize: 11 },
  },

  legend: {
    textStyle:    { color: '#64748b', fontSize: 11 },
    inactiveColor: '#e2e8f0',
  },

  tooltip: {
    backgroundColor: '#ffffff',
    borderColor:     '#e2e8f0',
    borderWidth:     1,
    textStyle:       { color: '#1e293b', fontSize: 12 },
    extraCssText:    'box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-radius: 6px;',
  },

  axisPointer: {
    lineStyle:  { color: '#cbd5e1', width: 1 },
    crossStyle: { color: '#cbd5e1' },
    label:      { backgroundColor: '#f8fafc', color: '#64748b', borderColor: '#e2e8f0' },
  },

  toolbox: {
    iconStyle:   { borderColor: '#94a3b8' },
    emphasis: {
      iconStyle: { borderColor: '#2563eb' },
    },
  },

  dataZoom: [
    {
      backgroundColor:   '#f8fafc',
      dataBackground: {
        lineStyle:  { color: '#e2e8f0' },
        areaStyle:  { color: '#f1f5f9' },
      },
      fillerColor:     'rgba(37, 99, 235, 0.08)',
      handleStyle:     { color: '#2563eb', borderColor: '#1d4ed8' },
      moveHandleStyle: { color: '#2563eb' },
      textStyle:       { color: '#94a3b8' },
      borderColor:     '#e2e8f0',
    },
  ],

  categoryAxis: {
    axisLine:   { lineStyle: { color: '#e2e8f0' } },
    axisTick:   { lineStyle: { color: '#e2e8f0' } },
    axisLabel:  { color: '#94a3b8', fontSize: 11 },
    splitLine:  { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
    splitArea:  { areaStyle: { color: ['rgba(0,0,0,0.01)', 'transparent'] } },
  },

  valueAxis: {
    axisLine:   { lineStyle: { color: '#e2e8f0' } },
    axisTick:   { lineStyle: { color: '#e2e8f0' } },
    axisLabel:  { color: '#94a3b8', fontSize: 11 },
    splitLine:  { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
    splitArea:  { areaStyle: { color: ['rgba(0,0,0,0.01)', 'transparent'] } },
  },

  logAxis: {
    axisLine:   { lineStyle: { color: '#e2e8f0' } },
    axisTick:   { lineStyle: { color: '#e2e8f0' } },
    axisLabel:  { color: '#94a3b8', fontSize: 11 },
    splitLine:  { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
  },

  timeAxis: {
    axisLine:   { lineStyle: { color: '#e2e8f0' } },
    axisTick:   { lineStyle: { color: '#e2e8f0' } },
    axisLabel:  { color: '#94a3b8', fontSize: 11 },
    splitLine:  { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
  },

  grid: {
    borderColor: '#e2e8f0',
  },
}

// ── Registration ──────────────────────────────────────────────────────────────
export function registerDsbfThemes() {
  echarts.registerTheme('dsbf-dark',  DARK)
  echarts.registerTheme('dsbf-light', LIGHT)
}

// Export raw objects for use in option merging if needed
export { DARK as dsbfDarkTheme, LIGHT as dsbfLightTheme, PALETTE as dsbfPalette }
