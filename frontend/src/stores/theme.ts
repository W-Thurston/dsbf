import { defineStore } from 'pinia'
import { ref } from 'vue'

export type Theme = 'dark' | 'light'

/** The dashboard theme, which `DsbfChart` and the app shell follow. */
export const useThemeStore = defineStore('theme', () => {
  const theme = ref<Theme>('dark')

  function toggle() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }

  return { theme, toggle }
})
