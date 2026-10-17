import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import router from './router/index.js'
import { registerDsbfThemes } from './chartTheme.js'

// Register ECharts themes once at startup so every <v-chart> component
// can reference 'dsbf-dark' or 'dsbf-light' by name.
registerDsbfThemes()

createApp(App).use(router).mount('#app')
