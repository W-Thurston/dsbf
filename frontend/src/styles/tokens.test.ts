import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const css = readFileSync(new URL('./tokens.css', import.meta.url), 'utf8')

function tokens(selector: string): Record<string, string> {
  const block = css.split(`${selector} {`)[1]?.split('}')[0] ?? ''
  return Object.fromEntries(
    [...block.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6});/g)].map(([, name, value]) => [name, value]),
  ) as Record<string, string>
}

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const channel = parseInt(hex.slice(i, i + 2), 16) / 255
    return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
  }) as [number, number, number]
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrast(a: string, b: string): number {
  const [light, dark] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number]
  return (light + 0.05) / (dark + 0.05)
}

const PAIRS: [string, string][] = [
  ['text', 'bg'],
  ['text', 'surface'],
  ['text-muted', 'bg'],
  ['text-muted', 'surface'],
  ['link', 'bg'],
  ['link', 'surface'],
  ['link-hover', 'surface'],
  ['error-fg', 'error-bg'],
  ['warn-fg', 'warn-bg'],
  ['notice-fg', 'notice-bg'],
  ['info-fg', 'info-bg'],
  ['ok-fg', 'ok-bg'],
  ['error-fg', 'surface'],
]

describe.each([['body'], ['body.theme-light']])('the %s theme tokens', (selector) => {
  const theme = { ...tokens('body'), ...tokens(selector) }

  it.each(PAIRS)('%s on %s reaches WCAG AA (4.5:1)', (foreground, background) => {
    expect(theme[foreground], foreground).toBeDefined()
    expect(theme[background], background).toBeDefined()
    expect(contrast(theme[foreground]!, theme[background]!)).toBeGreaterThanOrEqual(4.5)
  })
})
