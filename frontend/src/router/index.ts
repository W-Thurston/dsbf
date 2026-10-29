import { createRouter, createWebHistory } from 'vue-router'

// One lazy chunk per route; the dashboard's views are added unit by unit. A dataset key
// is a path or a name, so it travels percent-encoded in a single path segment.
export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'home', component: () => import('../views/HomeView.vue') },
    {
      path: '/datasets/:datasetKey',
      name: 'dataset',
      component: () => import('../views/DatasetView.vue'),
      props: true,
    },
    { path: '/compare', name: 'compare', component: () => import('../views/CompareView.vue') },
    {
      path: '/runs/:runId',
      name: 'run',
      component: () => import('../views/RunView.vue'),
      props: true,
    },
    {
      path: '/runs/:runId/findings',
      name: 'findings',
      component: () => import('../views/FindingsView.vue'),
      props: true,
    },
    {
      path: '/runs/:runId/columns',
      name: 'columns',
      component: () => import('../views/ColumnsView.vue'),
      props: true,
    },
    {
      path: '/runs/:runId/relationships',
      name: 'relationships',
      component: () => import('../views/RelationshipsView.vue'),
      props: true,
    },
    {
      path: '/runs/:runId/quality',
      name: 'quality',
      component: () => import('../views/QualityView.vue'),
      props: true,
    },
    {
      path: '/runs/:runId/time-series',
      name: 'time-series',
      component: () => import('../views/TimeSeriesView.vue'),
      props: true,
    },
    {
      path: '/runs/:runId/columns/:column',
      name: 'column',
      component: () => import('../views/ColumnView.vue'),
      props: true,
    },
  ],
})
