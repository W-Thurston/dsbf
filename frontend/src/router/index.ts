import { createRouter, createWebHistory } from 'vue-router'

// One lazy chunk per route. A dataset key is a path or a name, so it travels
// percent-encoded in a single path segment. A run's pages share one shell (header,
// health bar, tabs); `meta.tab` names the tab a page belongs to, and a page without one
// (the findings explorer, one column) leaves every tab inactive.
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
      component: () => import('../views/RunShell.vue'),
      props: true,
      children: [
        {
          path: '',
          name: 'run',
          component: () => import('../views/RunView.vue'),
          props: true,
          meta: { tab: 'overview' },
        },
        {
          path: 'quality',
          name: 'quality',
          component: () => import('../views/QualityView.vue'),
          props: true,
          meta: { tab: 'quality' },
        },
        {
          path: 'distributions',
          name: 'columns',
          component: () => import('../views/ColumnsView.vue'),
          props: true,
          meta: { tab: 'distributions' },
        },
        // The Phase 6 address of the column list.
        {
          path: 'columns',
          redirect: (to) => ({ name: 'columns', params: to.params, query: to.query }),
        },
        {
          path: 'columns/:column',
          name: 'column',
          component: () => import('../views/ColumnView.vue'),
          props: true,
          meta: { tab: 'distributions' },
        },
        {
          path: 'relationships',
          name: 'relationships',
          component: () => import('../views/RelationshipsView.vue'),
          props: true,
          meta: { tab: 'relationships' },
        },
        {
          path: 'ml-readiness',
          name: 'ml-readiness',
          component: () => import('../views/MlReadinessView.vue'),
          props: true,
          meta: { tab: 'ml-readiness' },
        },
        {
          path: 'time-series',
          name: 'time-series',
          component: () => import('../views/TimeSeriesView.vue'),
          props: true,
          meta: { tab: 'time-series' },
        },
        {
          path: 'findings',
          name: 'findings',
          component: () => import('../views/FindingsView.vue'),
          props: true,
        },
      ],
    },
  ],
})
