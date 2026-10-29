// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { parseFindingQuery } from '../../composables/findingQuery'
import ArtifactViewer from './ArtifactViewer.vue'
import FindingFilters from './FindingFilters.vue'

const get = vi.hoisted(() => vi.fn())
vi.mock('../../api/client', async (original) => ({
  ...(await original<typeof import('../../api/client')>()),
  api: { GET: get },
}))

afterEach(() => get.mockReset())

describe('FindingFilters', () => {
  it('emits the chosen filters from the first page and closes the open finding', async () => {
    const query = parseFindingQuery({ offset: '50', finding: 'f1' })
    const form = mount(FindingFilters, { props: { query, dimensions: ['leakage', 'validity'] } })

    await form.find('select').setValue('leakage')
    await form.find('input[type=text]').setValue(' null_rates ')
    await form.find('form').trigger('submit')

    const [[emitted]] = form.emitted('change') as [[Record<string, unknown>]]
    expect(emitted).toMatchObject({
      dimension: 'leakage',
      check: 'null_rates',
      offset: 0,
      finding: '',
    })
  })
})

describe('ArtifactViewer', () => {
  const rows = (offset: number) => ({
    columns: ['a', 'b'],
    rows: [[1, 'x']],
    offset,
    limit: 20,
    total: 45,
    ref: 'r',
    sampling: null,
  })

  it('loads nothing until asked, then pages by row range', async () => {
    const calls: { offset: number; limit: number }[] = []
    get.mockImplementation(
      async (_path: string, init: { params: { query: (typeof calls)[0] } }) => {
        calls.push(init.params.query)
        return { data: rows(init.params.query.offset), response: new Response() }
      },
    )
    const viewer = mount(ArtifactViewer, {
      props: { runId: 'run', artifactRef: 'pairs/fuzzy.parquet' },
    })
    expect(calls).toEqual([])

    await viewer.find('button').trigger('click')
    await flushPromises()
    expect(viewer.find('tbody').text()).toContain('x')
    expect(calls).toEqual([{ offset: 0, limit: 20 }])

    await viewer
      .findAll('button')
      .find((b) => b.text() === 'Next')!
      .trigger('click')
    await flushPromises()
    expect(calls[1]).toEqual({ offset: 20, limit: 20 })
  })
})
