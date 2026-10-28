<template>
  <header class="app-header">
    <router-link to="/">DSBF</router-link>
    <button class="theme-toggle" type="button" @click="themeStore.toggle()">
      {{ themeStore.theme === 'dark' ? 'Light' : 'Dark' }} theme
    </button>
  </header>
  <main>
    <router-view />
  </main>
</template>

<script setup lang="ts">
import { watchEffect } from 'vue'
import { useThemeStore } from './stores/theme'

const themeStore = useThemeStore()
watchEffect(() => document.body.classList.toggle('theme-light', themeStore.theme === 'light'))
</script>

<style>
.app-header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  justify-content: space-between;
}

.stack {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

main {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}
</style>
