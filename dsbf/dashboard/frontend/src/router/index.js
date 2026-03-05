/**
 * src/router/index.js
 *
 * Three routes:
* Three routes:
 *   /                          → DatasetListView  (pick a dataset)
 *   /datasets/:name            → RunListView      (pick a run)
 *   /datasets/:name/runs/:key  → RunDetailView    (the dashboard)
 */

import { createRouter, createWebHistory } from 'vue-router'
import DatasetListView from '../views/DatasetListView.vue'
import RunListView     from '../views/RunListView.vue'
import RunDetailView   from '../views/RunDetailView.vue'

const routes = [
  {
    path: '/',
    name: 'datasets',
    component: DatasetListView,
  },
  {
    path: '/datasets/:name',
    name: 'runs',
    component: RunListView,
    props: true,
  },
  {
    path: '/datasets/:name/runs/:runKey',
    name: 'run-detail',
    component: RunDetailView,
    props: true,
  },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
