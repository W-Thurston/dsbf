import { createRouter, createWebHistory } from 'vue-router'

// One lazy chunk per route; the dashboard's views are added unit by unit.
export default createRouter({
  history: createWebHistory(),
  routes: [{ path: '/', name: 'home', component: () => import('../views/HomeView.vue') }],
})
