/**
 * src/utils.js
 *
 * Shared utility functions used across multiple components.
 * Import from here rather than duplicating logic in each file.
 */

// ── Date formatting ───────────────────────────────────────────────────────────

/**
 * Format an ISO datetime string for display in the meta strip.
 * Produces a compact form: "Mar 3 '26, 07:53"
 */
export function formatDate(iso) {
  if (!iso) return '-'
  const d    = new Date(iso)
  const date = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  const yr   = String(d.getFullYear()).slice(2)
  const time = d.toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', hour12: false,
  })
  return `${date} '${yr}, ${time}`
}

/**
 * Format an ISO datetime string as a full human-readable timestamp.
 * Produces: "Mar 3, 2026, 07:53:08 AM"
 */
export function formatDateFull(iso) {
  if (!iso) return '-'
  return new Date(iso).toLocaleString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

// ── Quality score helpers ─────────────────────────────────────────────────────

/**
 * Return a CSS class name based on a 0–100 quality score.
 * Classes are defined in App.vue global styles.
 */
export function qualityClass(score) {
  if (score == null) return ''
  if (score >= 90) return 'score-excellent'
  if (score >= 70) return 'score-good'
  if (score >= 40) return 'score-warn'
  return 'score-poor'
}

/**
 * Return a plain-English label for a quality score.
 */
export function qualityLabel(score) {
  if (score == null) return ''
  if (score >= 90) return 'Excellent Quality'
  if (score >= 70) return 'Good Quality'
  if (score >= 40) return 'Fair Quality'
  return 'Poor Quality'
}

// ── String helpers ───────────────────────────────────────────────────────────

/**
 * Truncate a string to maxLen characters, appending '…' if truncated.
 * Safe to call with null/undefined - returns '-' in that case.
 */
export function trunc(val, maxLen = 20) {
  if (val == null) return '-'
  const s = String(val)
  return s.length > maxLen ? s.slice(0, maxLen) + '…' : s
}

// ── Figure helpers ────────────────────────────────────────────────────────────

/**
 * Find the best matching figure from a figures array for a given plot type,
 * format, and theme. Falls back to theme="default" for theme-less figures
 * (e.g. missingness_matrix).
 *
 * @param {Array}  figures  - Array of figure records from the API
 * @param {string} plotType - e.g. "correlation_matrix"
 * @param {string} format   - "interactive" | "static"
 * @param {string} theme    - "dark" | "light"
 * @returns {Object|null}
 */
export function figureFor(figures, plotType, format, theme) {
  const candidates = figures.filter(
    f => f.plot_type   === plotType
      && f.format      === format
      && f.column_name === null
  )
  return (
    candidates.find(f => f.theme === theme)     ??
    candidates.find(f => f.theme === 'default') ??
    null
  )
}
