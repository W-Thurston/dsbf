import { createApp } from 'vue'
import { createPinia } from 'pinia'
import './style.css'
import './styles/cards.css'
import App from './App.vue'
import router from './router'
import { registerDsbfThemes } from './chartTheme'

// Register the ECharts themes once so every chart can name 'dsbf-dark' or 'dsbf-light'.
registerDsbfThemes()

createApp(App).use(createPinia()).use(router).mount('#app')
