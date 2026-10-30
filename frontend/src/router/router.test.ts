// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'
import router from '.'

describe('the run routes', () => {
  it('keeps the Phase 6 addresses working', async () => {
    await router.push('/runs/abc/columns?type=numeric')
    expect(router.currentRoute.value.path).toBe('/runs/abc/distributions')
    expect(router.currentRoute.value.query).toEqual({ type: 'numeric' })

    await router.push('/runs/abc/columns/Age')
    expect(router.currentRoute.value.name).toBe('column')
    expect(router.currentRoute.value.meta.tab).toBe('distributions')
  })

  it('puts every tab under the run shell and leaves findings tab-less', async () => {
    for (const [path, tab] of [
      ['', 'overview'],
      ['/quality', 'quality'],
      ['/distributions', 'distributions'],
      ['/relationships', 'relationships'],
      ['/ml-readiness', 'ml-readiness'],
      ['/time-series', 'time-series'],
    ] as const) {
      await router.push(`/runs/abc${path}`)
      expect(router.currentRoute.value.meta.tab, path).toBe(tab)
      expect(router.currentRoute.value.matched).toHaveLength(2)
    }
    await router.push('/runs/abc/findings')
    expect(router.currentRoute.value.meta.tab).toBeUndefined()
  })
})
