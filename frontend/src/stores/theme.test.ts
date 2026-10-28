import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useThemeStore } from './theme'

describe('the theme store', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('starts dark and toggles to light and back', () => {
    const store = useThemeStore()
    expect(store.theme).toBe('dark')
    store.toggle()
    expect(store.theme).toBe('light')
    store.toggle()
    expect(store.theme).toBe('dark')
  })
})
